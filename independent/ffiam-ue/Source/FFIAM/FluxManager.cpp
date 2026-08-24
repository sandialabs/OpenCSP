// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
// ReSharper disable CppUE4CodingStandardNamingViolationWarning

#include "FluxManager.h"
#include "GenericPlatform/GenericPlatformProcess.h"

#include "Components/InstancedStaticMeshComponent.h"
#include "Kismet/KismetMathLibrary.h"
#include "Materials/MaterialInterface.h"

#pragma warning(disable: 4800)  // implicit conversion int to bool

#include "IrradianceFunctionLibrary.h"
#include "IrradianceMode.h"
#include "LogChannels.h"
#include "SiteConfigTypes.h"

// Voxel indexing + grid dimensions, straight from the library that writes Irrads.
#include "core/voxel.h"

#include <algorithm>

#define M_TO_CM 100.0f


namespace
{
	// The one place the ENU -> Unreal convention lives.
	//
	// The library hands back a voxel's MINIMUM corner in metres, (x,y,z)=(east,
	// north,up). Unreal is centimetres with X flipped (west-north-up). The
	// Shape_Flux_*_1m pivot is the centre of the cube's lower face -- bounds
	// -50..50 in X/Y, 0..100 in Z -- so an instance needs half its extent added in
	// X and Y to sit on the cell it represents, while Z already grows upward from
	// the pivot and must NOT be shifted.
	FVector EnuToUnrealCm(const float3& CornerMetres, const int32 ExtentMetres)
	{
		const float Half = 0.5f * static_cast<float>(ExtentMetres);
		return FVector(-(CornerMetres.x + Half),
		                 CornerMetres.y + Half,
		                 CornerMetres.z) * M_TO_CM;
	}
}


AFluxManager::AFluxManager()
{
	PrimaryActorTick.bCanEverTick = false;
	RootMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("RootMesh"));
	RootComponent = RootMesh;

	VoxelHism = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("Cube_AirHigh"));
	HelioHism = CreateDefaultSubobject<UInstancedStaticMeshComponent>(TEXT("Heliostat"));

	VoxelHism->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
	HelioHism->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);

	// --- Component invariants.
	//
	// These are contract properties, not tunables, so the code owns them -- but they
	// belong HERE rather than in BeginPlay. Set in the constructor they become CDO
	// defaults: correct in the editor viewport, correct for derived Blueprints, and
	// visible to whoever opens BP_FluxManager. Set in BeginPlay the Blueprint keeps
	// displaying wrong values forever and gets silently stomped at runtime.

	// Both components are rebuilt from scratch every analysis (ClearInstances +
	// a bulk refill), so they are not static geometry whatever the Blueprint says.
	// Static mobility lets the renderer cache visibility and draw state on the
	// assumption the primitive never changes, which misbehaves when the instance set
	// is replaced at runtime.
	VoxelHism->SetMobility(EComponentMobility::Movable);
	HelioHism->SetMobility(EComponentMobility::Movable);

	// Translucent shadow casting is expensive and meaningless for a flux cloud --
	// it is an irradiance readout, not an object that occludes light.
	VoxelHism->SetCastShadow(false);

	// The voxel ISM must sit at identity scale. Shape_Flux_*_1m are authored as 1 m
	// cubes, so an instance scale of N draws an N-metre voxel directly; any
	// component-level scale would multiply both instance size AND instance
	// translation, placing voxels at the wrong world position.
	VoxelHism->SetRelativeScale3D(FVector::OneVector);

	// One voxel component (TODO item 5): there is no second translucent primitive to
	// order against, so the per-primitive translucency-sort workaround is gone.

	AimPointComponent = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("AimPoint"));
	AimPointComponent->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);

	// TallTower = CreateDefaultSubobject<AActor>(TEXT("TallTower"));
	// NsttfTower = CreateDefaultSubobject<AActor>(TEXT("NsttfTower"));

}


AFluxManager::~AFluxManager()
{
}


void AFluxManager::BeginPlay()
{
	Super::BeginPlay();

	// Every instance transform this actor emits is world-space, and the library's
	// coordinate system puts the tower at the origin. A non-identity actor transform
	// therefore buys nothing -- but it also does no harm now that every AddInstance
	// passes bWorldSpace=true, so the code is genuinely transform-independent.
	// Report it rather than silently rewriting level-authored state at runtime.
	if (!GetActorTransform().Equals(FTransform::Identity))
	{
		UE_LOG(LogFlux, Warning,
			TEXT("FluxManager placed at %s in the level, not the origin. Instances are "
			     "world-space so rendering is unaffected, but component-relative values "
			     "(bounds origin, anything switched to local space) will read shifted. "
			     "Move the actor to the origin to silence this."),
			*GetActorTransform().GetLocation().ToCompactString());
	}

	SetSite(0);

	Threshold1 = 2.0f;
	Threshold2 = 4.0f;

	// Prep UE meshes
	checkf(VoxelMesh, TEXT("Mesh not set"));
	VoxelHism->SetStaticMesh(VoxelMesh);
	VoxelHism->ClearInstances();

	// Component invariants (mobility, cast shadow, voxel scale) are constructor-set
	// CDO defaults -- BP_FluxManager no longer overrides any of them, so nothing is
	// corrected here. Do not reintroduce runtime writes: that is what let the
	// Blueprint display wrong values indefinitely while being stomped on every run.

	UE_LOG(LogTemp, Warning, TEXT("Loading heliostat mesh"));
	checkf(HelioMesh, TEXT("Mesh not set"));
	HelioHism->SetStaticMesh(HelioMesh);
	HelioHism->SetRelativeScale3D(FVector(HelioSide));
	HelioHism->ClearInstances();
	UE_LOG(LogTemp, Warning, TEXT("Scaling heliostats by %.1f"), HelioSide);

	AnalysisDone = false;
}


void AFluxManager::DoAnalysisAsync()
{
	// Prevent multiple concurrent analyses
	if (bIsAnalyzing)
	{
		UE_LOG(LogFlux, Warning, TEXT("Analysis already in progress"));
		return;
	}

	UE_LOG(LogFlux, Warning, TEXT("FluxManager: Beginning CUDA analysis (async)"));

	auto Mode = UIrradianceFunctionLibrary::GetGameMode(this);

	// Subscribe to the completion delegate
	if (!Mode->OnAnalysisComplete.IsAlreadyBound(this, &ThisClass::OnAnalysisCompleted))
	{
		Mode->OnAnalysisComplete.AddDynamic(this, &ThisClass::OnAnalysisCompleted);
	}

	bIsAnalyzing = true;

	// Start async analysis
	Mode->DoAnalysisAsync(CspSite, Field, SelectedSiteConfig.HelioDesign, AimStrat, Datetime);
}


void AFluxManager::OnAnalysisCompleted(const bool bSuccess)
{
	bIsAnalyzing = false;

	if (!bSuccess)
	{
		UE_LOG(LogFlux, Error, TEXT("FluxManager: async analysis failed!"));
		return;
	}

	UE_LOG(LogFlux, Warning, TEXT("FluxManager: async analysis completed successfully"));

	// Get the results from the Mode
	const AIrradianceMode* Mode = UIrradianceFunctionLibrary::GetGameMode(this);
	const FAnalysisResult OutResult = Mode->GetLastAnalysisResult();

	// Process the results (same code as before, but now running on game thread after async completion)
	float HelioVerticalOffset = 1.0f * M_TO_CM / 2.0f;
	
	if (CspSite == ECspSite::Site_Nsttf)
	{
		HelioVerticalOffset = -1.0f * M_TO_CM / 2.0f;
		NsttfTower->SetActorLocation(FVector(0,0,-120.f));
		TallTower->SetActorLocation(FVector(0,0,-20000.f));
	}
	else
	{
		TallTower->SetActorLocation(FVector(0,0,-120.f));
		NsttfTower->SetActorLocation(FVector(0,0,-6000.f));
	}

	float IrradHighThreshold = Threshold2; // kW/m2
	float IrradMedThreshold = Threshold1;

	if (AnalysisDone)
	{
		HelioHism->ClearInstances();
		VoxelHism->ClearInstances();
		FlushPersistentDebugLines(GetWorld());
	}
	AnalysisDone = false;

	// Draw aim-point
	if (AimStrat.AimStrategyType == EAimStrategyType::Aim_Point)
	{
		FVector ConvertedAim;
		ConvertedAim.X = M_TO_CM * AimStrat.Params.X * -1.f; // ENU to WNU
		ConvertedAim.Y = M_TO_CM * AimStrat.Params.Y;
		ConvertedAim.Z = M_TO_CM * AimStrat.Params.Z;

		AimPointComponent->SetHiddenInGame(false);
		AimPointComponent->SetWorldTransform(FTransform(ConvertedAim));
		UE_LOG(LogFlux, Warning, TEXT("Aim: (%.2f, %.2f, %.2f), cm"), ConvertedAim.X, ConvertedAim.Y, ConvertedAim.Z);
	}
	else
	{
		AimPointComponent->SetHiddenInGame(true);
		AimPointComponent->SetWorldTransform(FTransform(FVector(0, 0, -2000)));
	}

	// --- Render voxels: shell only.
	// A solid mass of translucent cubes only ever shows its outer surface, so the
	// interior is pure overdraw. Keep a voxel only where it borders a LOWER class
	// (or the edge of the grid): that is the surface you can actually see, and it
	// cuts instance counts by roughly an order of magnitude. With the interior gone
	// the old detail-zone / merged-block LOD scheme is unnecessary -- everything is
	// drawn at native voxel size.
	TotalIrrad = 0.0f;
	TotalThresIrrad = 0.0f;
	NumIrradVoxels = 0;
	NumThresholdVoxels = 0;

	const int Vs = Field.VoxelSize;

	int NumVoxelsX = 0, NumVoxelsY = 0, NumVoxelsZ = 0;
	const int GridVoxelCount = ComputeVoxelGridDimensions(
		Field.Radius, Field.MinHeight, Field.MaxHeight, Vs, &NumVoxelsX, &NumVoxelsY, &NumVoxelsZ);

	// Stride between adjacent Y rows is the X count; a whole Z plane is X*Y. Square
	// in practice (both derived from Radius), but named for what they index.
	const int VoxelsPerPlane = NumVoxelsX * NumVoxelsY;

	// Field.NumVoxels sized the Irrads allocation; the loops below index it out to
	// GridVoxelCount. If those ever disagree the shell pass reads out of bounds, so
	// bail rather than render garbage from unmapped memory.
	if (GridVoxelCount != Field.NumVoxels)
	{
		UE_LOG(LogFlux, Error,
			TEXT("Voxel grid mismatch: library says %d voxels (%dx%dx%d) but Field.NumVoxels is %d. "
			     "Skipping voxel render -- the grid convention has drifted."),
			GridVoxelCount, NumVoxelsX, NumVoxelsY, NumVoxelsZ, Field.NumVoxels);

		// Still report completion, or the UI waits on a delegate that never fires.
		OnProcessingComplete.Broadcast(false);
		return;
	}

	// Occupancy per voxel: 1 = at or above the display threshold, 0 = below. Collapsed
	// from the old 0/med/high three-way split (TODO item 5): the colour ramp now carries
	// the flux magnitude continuously, so the med/high *bucket* no longer drives geometry
	// and everything above Threshold1 renders into one component. Threshold2 survives only
	// as a reporting/legend number (NumHighVoxels below). Uninitialized, not zeroed: the
	// loop assigns every element, and a zero-fill of the full grid is ~788 MB of pointless
	// memset on the largest sites.
	TArray<uint8> Occupied;
	Occupied.SetNumUninitialized(Field.NumVoxels);

	// >= Threshold2. No longer a rendering class -- reported for the legend/high-flux
	// readout only, so the second threshold still means something to the user.
	int NumHighVoxels = 0;

	for (int vi = 0; vi < Field.NumVoxels; ++vi)
	{
		const float Irrad = OutResult.Irrads[vi];
		if (Irrad > 0.0f) ++NumIrradVoxels;
		TotalIrrad += Irrad;

		if (Irrad >= IrradMedThreshold)
		{
			TotalThresIrrad += Irrad;
			++NumThresholdVoxels;
			if (Irrad >= IrradHighThreshold) ++NumHighVoxels;
			Occupied[vi] = 1;
		}
		else
		{
			Occupied[vi] = 0;
		}
	}

	// Shell extraction. With one occupancy class the rule collapses to: an occupied
	// voxel is visible if any of its six face-neighbours is empty, or it sits on the
	// grid boundary -- i.e. the outer surface of the whole above-threshold cloud. The
	// old lower-class comparison existed to draw the high/med interface once; there is
	// no interface now, so a plain occupied/empty test replaces it. This also removes
	// the nested med shell that used to enclose the high core -- the veiling that no
	// draw-order or opacity tuning could fix (TODO item 5).
	//
	// Trade-off: a genuinely enclosed hot core is now interior and culled, which is
	// acceptable only because the above-threshold region is thin beam tubes (nearly all
	// surface -- see item 6), so almost nothing is lost; verify this holds on the host.
	// Greedy 2D meshing (XY-rectangle merge, per Z-slice). Each drawn instance is a
	// rectangle: a maximal W x H block of visible shell voxels in one z-slice sharing an
	// irradiance band, emitted as one stretched box (W*Vs x H*Vs x Vs). This generalises
	// the earlier 1D X-run merge (which is the H==1 case) and additionally collapses flat
	// surfaces -- especially horizontal caps -- into far fewer instances, which is what
	// drives the render-thread / camera-movement cost at 1 m. Interior voxels are still
	// culled (they are not visible, so never seed or extend a rectangle); geometry is
	// identical to the individual cubes, only colour flattens to the rectangle's hottest
	// voxel, and only within MergeIrradRatio.
	struct FShellSpan { int32 Index; int32 W; int32 H; float Irrad; };  // Index = min-corner voxel
	TArray<FShellSpan> Shell;
	Shell.Reserve(NumThresholdVoxels);  // upper bound: at most one rectangle per voxel

	int32 ShellVoxelCount = 0;  // total surface voxels (pre-merge), for honest reporting

	// Irradiance -> integer band: two voxels merge only if their band matches, so a merged
	// box spans at most MergeIrradRatio in irradiance. Guard the log base so ratio==1 gives
	// effectively per-value bands (no cross-value merging).
	const float InvLogRatio = 1.0f / FMath::Loge(FMath::Max(MergeIrradRatio, 1.0f + KINDA_SMALL_NUMBER));

	auto VisibleAt = [&](int x, int y, int z) -> bool
	{
		const int vi = x + y * NumVoxelsX + z * VoxelsPerPlane;
		if (!Occupied[vi]) return false;
		// Boundary terms first: they guard the neighbour reads below (short-circuit).
		const bool bEdgeYZ = (z == 0) || (z == NumVoxelsZ - 1) || (y == 0) || (y == NumVoxelsY - 1);
		return bEdgeYZ || (x == 0) || (x == NumVoxelsX - 1) ||
		       !Occupied[vi - 1]              || !Occupied[vi + 1]              ||
		       !Occupied[vi - NumVoxelsX]     || !Occupied[vi + NumVoxelsX]     ||
		       !Occupied[vi - VoxelsPerPlane] || !Occupied[vi + VoxelsPerPlane];
	};
	auto BandAt = [&](int x, int y, int z) -> int32
	{
		const float Irr = OutResult.Irrads[x + y * NumVoxelsX + z * VoxelsPerPlane];
		return FMath::FloorToInt(FMath::Loge(FMath::Max(Irr, KINDA_SMALL_NUMBER)) * InvLogRatio);
	};

	// "Consumed this slice" via a generation stamp (== z), so no per-slice clear is needed:
	// a cell belongs to an already-emitted rectangle iff UsedStamp[cell] == z.
	TArray<int32> UsedStamp;
	UsedStamp.Init(-1, NumVoxelsX * NumVoxelsY);

	for (int z = 0; z < NumVoxelsZ; ++z)
	{
		for (int y0 = 0; y0 < NumVoxelsY; ++y0)
		{
			for (int x0 = 0; x0 < NumVoxelsX; ++x0)
			{
				const int Cell0 = x0 + y0 * NumVoxelsX;
				if (UsedStamp[Cell0] == z || !VisibleAt(x0, y0, z)) continue;

				const int32 Band = BandAt(x0, y0, z);

				// Grow width along X while visible, unconsumed, same band.
				int32 W = 1;
				while (x0 + W < NumVoxelsX)
				{
					const int c = Cell0 + W;
					if (UsedStamp[c] == z || !VisibleAt(x0 + W, y0, z) || BandAt(x0 + W, y0, z) != Band) break;
					++W;
				}

				// Grow height along Y while every cell of the full W-wide row qualifies.
				int32 H = 1;
				for (bool bGrow = true; bGrow && (y0 + H) < NumVoxelsY; )
				{
					for (int dx = 0; dx < W; ++dx)
					{
						const int c = (x0 + dx) + (y0 + H) * NumVoxelsX;
						if (UsedStamp[c] == z || !VisibleAt(x0 + dx, y0 + H, z) || BandAt(x0 + dx, y0 + H, z) != Band)
						{
							bGrow = false;
							break;
						}
					}
					if (bGrow) ++H;
				}

				// Claim the rectangle and colour it by its hottest voxel.
				float MaxIrr = 0.0f;
				for (int dy = 0; dy < H; ++dy)
				{
					for (int dx = 0; dx < W; ++dx)
					{
						const int c = (x0 + dx) + (y0 + dy) * NumVoxelsX;
						UsedStamp[c] = z;
						MaxIrr = FMath::Max(MaxIrr, OutResult.Irrads[c + z * VoxelsPerPlane]);
					}
				}

				Shell.Add({Cell0 + z * VoxelsPerPlane, W, H, MaxIrr});
				ShellVoxelCount += W * H;
			}
		}
	}

	// The sort exists only to decide who loses to the cap, so skip the n log n
	// entirely when nothing will be dropped.
	//
	// NB: shell extraction does NOT reliably cut counts by an order of magnitude, as
	// was assumed when the cap was sized. Measured on test_radial_sm (1 km radial,
	// 2 m voxels, 6400 helios): 4,275,354 shell of 5,784,081 above threshold -- a 26%
	// reduction. The above-threshold region is a bundle of thin beam tubes rather than
	// a compact blob, and a tube only a few voxels across is nearly all surface, so
	// there is little interior to remove. Expect the cap to bind on large sites.
	const int32 ShellCount = Shell.Num();  // spans (render instances), not voxels
	const int32 DrawCount = FMath::Min(ShellCount, MaxVoxelInstances);

	// HUD readout: total shell voxels (drawn voxels summed over spans after the cap, below).
	LastShellVoxelCount = ShellVoxelCount;

	if (NumThresholdVoxels > 0)
	{
		UE_LOG(LogFlux, Display,
			TEXT("Shell extraction: %d of %d above-threshold voxels kept (%.1f%% culled as interior); "
			     "greedy-meshed to %d boxes (%.1fx fewer instances)"),
			ShellVoxelCount, NumThresholdVoxels,
			100.0f * (1.0f - static_cast<float>(ShellVoxelCount) / static_cast<float>(NumThresholdVoxels)),
			ShellCount,
			ShellCount > 0 ? static_cast<float>(ShellVoxelCount) / static_cast<float>(ShellCount) : 1.0f);
	}

	if (DrawCount < ShellCount)
	{
		// Hottest first, so the faintest boxes are the ones that go.
		Shell.Sort([](const FShellSpan& A, const FShellSpan& B) { return A.Irrad > B.Irrad; });

		// Budget by raising the effective threshold rather than truncating silently:
		// keeping the top N of an irradiance-sorted list IS a higher threshold, so
		// report the cutoff instead of pretending everything is on screen.
		UE_LOG(LogFlux, Warning,
			TEXT("Voxel cap reached: showing %d of %d shell boxes; effective display threshold "
			     "raised %.2f -> %.2f kW/m2 (%d boxes dropped)"),
			DrawCount, ShellCount, IrradMedThreshold, Shell[DrawCount - 1].Irrad, ShellCount - DrawCount);
	}

	// Ramp top from a PERCENTILE of the drawn distribution, not its maximum.
	//
	// Flux is long-tailed: the focal spot runs two orders of magnitude above the
	// bulk, so normalising to the max hands most of the colour range to a handful of
	// voxels and crushes everything else onto the bottom of the ramp. Measured on
	// test_radial_sm: max 622.6 kW/m2 but a mean drawn value of ~4 kW/m2, leaving the
	// two shells only ~14% of the ramp apart and visually indistinguishable.
	//
	// Everything above the percentile clamps to the top colour, which is the correct
	// read for a focal spot anyway -- "at or above peak" rather than a precise value.
	LastShellRampTop = 1.0f;
	if (DrawCount > 0)
	{
		// Copy rather than partition Shell in place: the fill loop below depends on
		// its current order (hottest-first when the cap is binding).
		TArray<float> Drawn;
		Drawn.SetNumUninitialized(DrawCount);
		float DrawnMax = 0.0f;
		for (int32 i = 0; i < DrawCount; ++i)
		{
			Drawn[i] = Shell[i].Irrad;
			DrawnMax = FMath::Max(DrawnMax, Drawn[i]);
		}

		const int32 Idx = FMath::Clamp(
			FMath::FloorToInt(DrawCount * (ColorScalePercentile / 100.0f)), 0, DrawCount - 1);

		// nth_element is O(n) -- a full sort of 4M floats to read one of them would be
		// pure waste, and this runs on the game thread.
		std::nth_element(Drawn.GetData(), Drawn.GetData() + Idx, Drawn.GetData() + DrawCount);

		LastShellRampTop = FMath::Max(Drawn[Idx], Threshold1 * 1.01f);

		UE_LOG(LogFlux, Display,
			TEXT("Ramp top: p%.1f = %.2f kW/m2 (drawn max %.2f). Values above p%.1f clamp to the "
			     "top of the ramp."),
			ColorScalePercentile, LastShellRampTop, DrawnMax, ColorScalePercentile);
	}

	// Per-instance custom data carries normalized irradiance so the material can ramp
	// colour continuously. The mapping itself lives in ApplyColorScale so the
	// absolute/relative toggle can re-apply it without re-running the analysis.
	//
	// Use the setter, not the bare member: it resizes PerInstanceSMCustomData to
	// match. Assigning the field directly leaves that array unsized and
	// SetCustomDataValue then silently no-ops.
	VoxelHism->SetNumCustomDataFloats(1);

	VoxelInstanceIrrads.Reset(DrawCount);

	// Batched fill (TODO item 2). One component now (item 5): every occupied-shell voxel
	// goes into VoxelHism, distinguished only by ramp colour. AddInstance appends one at
	// a time and each call dirties the instance-update tracker and issues a
	// PartialNavigationUpdate -- at the 4M cap that is millions of redundant bookkeeping
	// passes. AddInstances does it in one pass per call.
	//
	// Chunk the staging array rather than building all DrawCount transforms at once:
	// FTransform is ~96 B under LWC, so a single 4M array would stage ~384 MB. 64k
	// transforms per flush caps the staging at ~6 MB while amortising the per-call
	// overhead to nothing. Instance index still equals VoxelInstanceIrrads index --
	// AddInstances appends in array order, and the batches are added front to back.
	constexpr int32 BatchSize = 64 * 1024;
	TArray<FTransform> Batch;
	Batch.Reserve(FMath::Min(DrawCount, BatchSize));

	int32 DrawnVoxels = 0;  // voxels covered by drawn spans, for the HUD readout

	for (int32 i = 0; i < DrawCount; ++i)
	{
		const FShellSpan& Sv = Shell[i];

		// A rectangle is W x H voxels in one z-slice. Place a box stretched to
		// (W*Vs, H*Vs, Vs) centred on it: the min- and max-corner voxels share z, and each
		// GetVoxelLocFromIndex->EnuToUnrealCm is that voxel's lower-face-centre, so their
		// midpoint is the rectangle's lower-face-centre and the X/Y-centred mesh pivot lands
		// it correctly. W==H==1 reduces to the original single-cube placement.
		const int32 OppIndex = Sv.Index + (Sv.W - 1) + (Sv.H - 1) * NumVoxelsX;
		const FVector A = EnuToUnrealCm(GetVoxelLocFromIndex(Sv.Index,  Vs, Field.Radius, Field.MinHeight), Vs);
		const FVector B = EnuToUnrealCm(GetVoxelLocFromIndex(OppIndex,  Vs, Field.Radius, Field.MinHeight), Vs);
		Batch.Add(FTransform(
			FRotator::ZeroRotator,
			(A + B) * 0.5f,
			FVector(Sv.W * Vs, Sv.H * Vs, Vs)));
		VoxelInstanceIrrads.Add(Sv.Irrad);   // index == instance index
		DrawnVoxels += Sv.W * Sv.H;

		if (Batch.Num() == BatchSize || i == DrawCount - 1)
		{
			VoxelHism->AddInstances(Batch, /*bShouldReturnIndices*/ false, /*bWorldSpace*/ true);
			Batch.Reset();   // Reset, not Empty: keep the reserved capacity for the next chunk.
		}
	}

	LastDrawnVoxelCount = DrawnVoxels;  // voxels on screen (spans may be capped)

	// Refresh the cached Bounds the renderer frustum-culls against. After ClearInstances
	// that cache is an empty box at the component's own location, so a stale bound makes
	// every instance vanish once that point leaves the frustum, however much geometry is
	// really there. The batched AddInstances path may already refresh bounds at the end
	// of each call; this is kept as a belt-and-braces guarantee until confirmed redundant
	// on the host, and is a cheap no-op if the bounds are already current. ApplyColorScale
	// marks the render state dirty after.
	VoxelHism->UpdateBounds();

	ApplyColorScale();

	// Blend-mode guard against a silent occlusion regression.
	//
	// The entire high-on-top scheme rests on BOTH voxel materials being translucent:
	// translucency does not write depth, so the two shells order purely by
	// TranslucentSortPriority and the enclosed high core shows through the med shell that
	// surrounds it. This project also ships M_AirFlux_Opaque / M_AirFlux_Med_Opaque, and
	// assigning either to the voxel mesh makes the component write depth -- at which
	// point nearer voxels hard-occlude farther ones instead of the cloud reading as a
	// translucent volume, and no compositing tweak can rescue it. That failure is
	// view-independent and leaves no trace in the existing logs, so name it here.
	// Silence on this line means any residual occlusion is NOT a depth-write.
	const UMaterialInterface* VoxelMat = VoxelHism ? VoxelHism->GetMaterial(0) : nullptr;
	if (VoxelMat)
	{
		const EBlendMode BM = VoxelMat->GetBlendMode();
		if (BM == BLEND_Opaque || BM == BLEND_Masked)
		{
			UE_LOG(LogFlux, Error,
				TEXT("Voxel material '%s' is opaque/masked, so it writes depth and nearer voxels "
				     "will hard-occlude farther ones. Assign the translucent ramp material "
				     "(M_AirFlux_Tran), not an _Opaque variant."),
				*VoxelMat->GetName());
		}
	}

	// Bounds sanity: this should enclose the rendered voxel cloud. A degenerate extent
	// means the refresh above did not take and culling will misbehave.
	const FBoxSphereBounds VoxelB = VoxelHism->Bounds;
	const FString BoundsMsg = FString::Printf(
		TEXT("Voxel ISM bounds: origin=%s extent=%s"),
		*VoxelB.Origin.ToCompactString(), *VoxelB.BoxExtent.ToCompactString());

	if (VoxelB.BoxExtent.IsNearlyZero())
	{
		UE_LOG(LogFlux, Warning, TEXT("%s  [DEGENERATE - voxels will be culled]"), *BoundsMsg);
	}
	else
	{
		UE_LOG(LogFlux, Display, TEXT("%s"), *BoundsMsg);
	}

	UE_LOG(LogFlux, Warning, TEXT("Total: %.2f kW; voxels with nonzero irrad: %d"), TotalIrrad, NumIrradVoxels);
	UE_LOG(LogFlux, Warning, TEXT("Flux Threshold 1: %.2f kW/m2 ; Flux Threshold 2: %.2f kW/m2"), IrradMedThreshold, IrradHighThreshold);
	UE_LOG(LogFlux, Warning,
		TEXT("Total irrad above threshold: %.2f kW (%d voxels, of which %d >= Threshold2)"),
		TotalThresIrrad, NumThresholdVoxels, NumHighVoxels);
	UE_LOG(LogFlux, Display,
		TEXT("Rendered: %d spans covering %d of %d shell voxels (of %d above threshold; instance cap %d)"),
		VoxelInstanceIrrads.Num(), LastDrawnVoxelCount, LastShellVoxelCount, NumThresholdVoxels, MaxVoxelInstances);

	RenderHeliostats(OutResult, HelioVerticalOffset);


	AnalysisDone = true;
	OnProcessingComplete.Broadcast(true);
	UE_LOG(LogFlux, Warning, TEXT("FluxManager: Rendering complete"));
}


void AFluxManager::RenderHeliostats(const FAnalysisResult& Result, const float HelioVerticalOffset)
{
	const FVector Up = FVector(0, 0, 1);
	UE_LOG(LogFlux, Warning, TEXT("Rendering heliostats"));

	for (int hi = 0; hi < Field.NumHelios; ++hi)
	{
		float3 loc = Result.Locs[hi];

		// Convert from ENU (east-north-up) to UE5 WNU (west-north-up) and to cm.
		loc.x *= M_TO_CM * -1.f;
		loc.y *= M_TO_CM;
		loc.z *= M_TO_CM;
		loc.z += HelioVerticalOffset;

		float3 aimV = Result.AimVs[hi];
		const FRotator RotNormal = UKismetMathLibrary::MakeRotFromXZ(FVector(-1.f * aimV.x, aimV.y, aimV.z), Up);
		const FTransform Trans = FTransform(RotNormal,
		                                    FVector(loc.x, loc.y, loc.z),
		                                    FVector(HelioSide, HelioSide, HelioSide));
		HelioHism->AddInstance(Trans, true);
	}

	// Same as the voxel ISMs: refresh the cached bounds the renderer culls against,
	// otherwise heliostats vanish wholesale at some viewpoints.
	HelioHism->UpdateBounds();
	HelioHism->MarkRenderStateDirty();

	UE_LOG(LogFlux, Warning, TEXT("%d heliostats rendered"), Field.NumHelios);
}


//
// --- Setters & Update Utilities
//
// Maps stored per-instance irradiance onto the material's 0-1 colour ramp input.
// Separate from the fill so the absolute/relative toggle can re-colour the existing
// instances instantly, with no re-analysis.
void AFluxManager::ApplyColorScale()
{
	if (!VoxelHism) return;

	// LOG scale, deliberately. Flux spans orders of magnitude -- the display
	// threshold is ~2 kW/m2 while the focal peak runs to five figures -- so a linear
	// map would collapse nearly every voxel onto the bottom of the ramp and leave a
	// handful of pixels carrying the whole colour range. The mapping lives in
	// GetRampLogRange/ApplyRampLogRange so the legend's tick marks share it exactly.
	float LogLo, InvLogRange, PeakKwM2;
	GetRampLogRange(LogLo, InvLogRange, PeakKwM2);

	int32 FailedWrites = 0;

	// One component now (TODO item 5): a single log-scaled ramp spans the whole drawn
	// distribution, from the Threshold1 outer surface up to the p-th percentile focal
	// spot. min/mean/max over that one set is the whole diagnostic -- if max sits near
	// the bottom of the range the ramp is being wasted (retune ColorScalePercentile),
	// and if it is near 1.0 the hot surface is reaching the top colour as intended.
	float MinT = 1.0f, MaxT = 0.0f, SumT = 0.0f;

	for (int32 i = 0; i < VoxelInstanceIrrads.Num(); ++i)
	{
		const float T = ApplyRampLogRange(VoxelInstanceIrrads[i], LogLo, InvLogRange);

		if (!VoxelHism->SetCustomDataValue(i, 0, T, /*bMarkRenderStateDirty*/ false))
		{
			++FailedWrites;
		}

		MinT = FMath::Min(MinT, T);
		MaxT = FMath::Max(MaxT, T);
		SumT += T;
	}
	VoxelHism->MarkRenderStateDirty();

	if (VoxelInstanceIrrads.Num() > 0)
	{
		UE_LOG(LogFlux, Warning,
			TEXT("Ramp input: %d instances, T min=%.3f mean=%.3f max=%.3f"),
			VoxelInstanceIrrads.Num(), MinT, SumT / VoxelInstanceIrrads.Num(), MaxT);
	}

	if (FailedWrites > 0)
	{
		UE_LOG(LogFlux, Error,
			TEXT("%d per-instance custom data writes FAILED -- the ramp input never reached the "
			     "material. Check SetNumCustomDataFloats."), FailedWrites);
	}

	// Named local, not an inline *FString::Printf(): UE_LOG expands to several
	// statements, so a temporary built inside the call can be destroyed before use.
	const FString ScaleMode = bAbsoluteColorScale
		? FString(TEXT("ABSOLUTE"))
		: FString::Printf(TEXT("relative to this run's p%.1f"), ColorScalePercentile);

	UE_LOG(LogFlux, Warning, TEXT("Colour ramp: %s, %.2f -> %.2f kW/m2 (log scale)"),
		*ScaleMode, Threshold1, PeakKwM2);
}


// The peak (per scale mode) and the log range derived from it. Threshold1 * 1.01 floors
// the peak so a run whose drawn max barely clears the threshold still has a non-degenerate
// range. This is the single place the ramp's endpoints are decided.
void AFluxManager::GetRampLogRange(float& OutLogLo, float& OutInvLogRange, float& OutPeakKwM2) const
{
	OutPeakKwM2 = bAbsoluteColorScale
		? FMath::Max(ColorScalePeak, Threshold1 * 1.01f)
		: FMath::Max(LastShellRampTop, Threshold1 * 1.01f);

	OutLogLo = FMath::Loge(FMath::Max(Threshold1, KINDA_SMALL_NUMBER));
	const float LogHi = FMath::Loge(OutPeakKwM2);
	OutInvLogRange = 1.0f / FMath::Max(LogHi - OutLogLo, KINDA_SMALL_NUMBER);
}


// Map one irradiance through a precomputed log range to T in [0,1]. Static + taking the
// range by value so the 4M-instance loop computes ln(Threshold1)/ln(peak) once, not per
// voxel, while RampInputFor can call it for a single legend tick.
float AFluxManager::ApplyRampLogRange(const float IrradKwM2, const float LogLo, const float InvLogRange)
{
	return FMath::Clamp(
		(FMath::Loge(FMath::Max(IrradKwM2, KINDA_SMALL_NUMBER)) - LogLo) * InvLogRange, 0.0f, 1.0f);
}


float AFluxManager::RampInputFor(const float IrradKwM2) const
{
	float LogLo, InvLogRange, PeakKwM2;
	GetRampLogRange(LogLo, InvLogRange, PeakKwM2);
	return ApplyRampLogRange(IrradKwM2, LogLo, InvLogRange);
}


void AFluxManager::SetColorScale(const bool bAbsolute, const float PeakKwM2)
{
	bAbsoluteColorScale = bAbsolute;
	ColorScalePeak = FMath::Max(PeakKwM2, KINDA_SMALL_NUMBER);

	// Re-colour in place; the geometry is unchanged so there is nothing to re-run.
	ApplyColorScale();
}


float AFluxManager::GetColorScalePeak() const
{
	return bAbsoluteColorScale ? ColorScalePeak : LastShellRampTop;
}


void AFluxManager::SetSite(const int Option)
{
	UE_LOG(LogTemp, Warning, TEXT("New site %d selected. Overwriting other inputs."), Option);

	const auto Mode = UIrradianceFunctionLibrary::GetGameMode(this);
	SelectedSiteConfig = Mode->GetSiteConfig(Option);
	CspSite = SelectedSiteConfig.SiteType;
	AimStrat = SelectedSiteConfig.AimStrategy;
	Datetime = SelectedSiteConfig.DateTime;
	Field = SelectedSiteConfig.Field;

	// Voxel size is per-site (JSON). Keep the render-side copy in lockstep,
	// otherwise a 1 m site still draws 2 m cubes.
	UE_LOG(LogFlux, Display, TEXT("Site voxel size: %d m"), Field.VoxelSize);
}


void AFluxManager::UpdateParameters(const int SiteIdx, const int NewYear, const int NewMonth, const int NewDay, const int NewHour)
{
	SetSite(SiteIdx);

	Datetime.Year = NewYear;
	Datetime.Month = NewMonth;
	Datetime.Day = NewDay;
	Datetime.Hour = NewHour;
}


void AFluxManager::SetDatetime(const int NewYear, const int NewMonth, const int NewDay, const int NewHour)
{
	Datetime.Year = NewYear;
	Datetime.Month = NewMonth;
	Datetime.Day = NewDay;
	Datetime.Hour = NewHour;
}


void AFluxManager::SetAim(const int AimIdx,
                          const float AimPtParam1, const float AimPtParam2, const float AimPtParam3,
                          const float AimRingParam1, const float AimRingParam2, const float AimRingParam3,
                          const float AimVecParam1, const float AimVecParam2, const float AimVecParam3)
{
	switch (AimIdx)
	{
	case 0:
		AimStrat.AimStrategyType = EAimStrategyType::Aim_Point;
		AimStrat.Params.X = AimPtParam1;
		AimStrat.Params.Y = AimPtParam2;
		AimStrat.Params.Z = AimPtParam3;
		break;
	case 1:
		// Ring params = (inner_radius, outer_radius, height)
		AimStrat.AimStrategyType = EAimStrategyType::Aim_Ring;
		AimStrat.Params.X = AimRingParam1;
		AimStrat.Params.Y = AimRingParam2;
		AimStrat.Params.Z = AimRingParam3;
		break;
	case 2:
		// Split Ring shares the ring convention: (inner_radius, outer_radius, height),
		// so it reuses the same ring input group in the HUD.
		AimStrat.AimStrategyType = EAimStrategyType::Aim_Split_Ring;
		AimStrat.Params.X = AimRingParam1;
		AimStrat.Params.Y = AimRingParam2;
		AimStrat.Params.Z = AimRingParam3;
		break;
	default:
		// AimIdx 3 = Vector
		AimStrat.AimStrategyType = EAimStrategyType::Aim_Vector;
		AimStrat.Params.X = AimVecParam1;
		AimStrat.Params.Y = AimVecParam2;
		AimStrat.Params.Z = AimVecParam3;
		break;
	}
}


void AFluxManager::SetThresholds(float Thres1, float Thres2)
{
	if (Thres1 >= 0.1f) Threshold1 = Thres1;

	if (Thres2 >= 1 && Thres2 > Thres1)
	{
		Threshold2 = Thres2;
	}
}
