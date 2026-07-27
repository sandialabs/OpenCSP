// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
// ReSharper disable CppUE4CodingStandardNamingViolationWarning

#include "FluxManager.h"
#include "GenericPlatform/GenericPlatformProcess.h"

#include "Components/HierarchicalInstancedStaticMeshComponent.h"
#include "Kismet/KismetMathLibrary.h"

#pragma warning(disable: 4800)  // implicit conversion int to bool

#include "IrradianceFunctionLibrary.h"
#include "IrradianceMode.h"
#include "LogChannels.h"
#include "SiteConfigTypes.h"

#include <algorithm>

#define M_TO_CM 100.0f


AFluxManager::AFluxManager()
{
	PrimaryActorTick.bCanEverTick = false;
	RootMesh = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("RootMesh"));
	RootComponent = RootMesh;

	VoxelHism = CreateDefaultSubobject<UHierarchicalInstancedStaticMeshComponent>(TEXT("Cube_AirHigh"));
	VoxelMedHism = CreateDefaultSubobject<UHierarchicalInstancedStaticMeshComponent>(TEXT("Cube_AirMed"));
	HelioHism = CreateDefaultSubobject<UHierarchicalInstancedStaticMeshComponent>(TEXT("Heliostat"));

	VoxelHism->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
	VoxelMedHism->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
	HelioHism->AttachToComponent(RootComponent, FAttachmentTransformRules::KeepRelativeTransform);
	
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

	SetSite(0);

	Threshold1 = 2.0f;
	Threshold2 = 4.0f;

	// Prep UE meshes
	checkf(VoxelMesh, TEXT("Mesh not set"));
	VoxelHism->SetStaticMesh(VoxelMesh);
	VoxelHism->SetRelativeScale3D(VoxelHism->GetRelativeScale3D() * VoxelSize);
	VoxelHism->ClearInstances();

	checkf(VoxelMedMesh, TEXT("Mesh not set"));
	VoxelMedHism->SetStaticMesh(VoxelMedMesh);
	VoxelMedHism->SetRelativeScale3D(VoxelMedHism->GetRelativeScale3D() * VoxelSize);
	VoxelMedHism->ClearInstances();

	UE_LOG(LogTemp, Warning, TEXT("Scaling voxel meshes by %d"), VoxelSize);

	UE_LOG(LogTemp, Warning, TEXT("Loading heliostat mesh"));
	checkf(HelioMesh, TEXT("Mesh not set"));
	HelioHism->SetStaticMesh(HelioMesh);
	HelioHism->SetRelativeScale3D(HelioHism->GetRelativeScale3D() * HelioSide);
	HelioHism->ClearInstances();
	UE_LOG(LogTemp, Warning, TEXT("Scaling heliostats by %.1f"), HelioSide);

	AnalysisDone = false;
}


bool AFluxManager::DoAnalysis()
{
	// Called when the user clicks "Analyze" in the UI.
	UE_LOG(LogFlux, Warning, TEXT("FluxManager: Beginning CUDA analysis"));

	auto Mode = UIrradianceFunctionLibrary::GetGameMode(this);

	FAnalysisResult OutResult;
	bool Status = Mode->DoAnalysis(CspSite, Field, SelectedSiteConfig.HelioDesign, AimStrat, Datetime, OutResult);

	if (!Status)
	{
		UE_LOG(LogFlux, Error, TEXT("FluxManager: analysis failed! Exiting."));
		return false;
	}

	float HelioVerticalOffset = 1.0f * M_TO_CM / 2.0f;
	if (CspSite == ECspSite::Site_Nsttf)
	{
		HelioVerticalOffset = -1.0f * M_TO_CM / 2.0f;
	}

	float IrradHighThreshold = Threshold2; // kW/m2
	float IrradMedThreshold = Threshold1;

	if (AnalysisDone)
	{
		HelioHism->ClearInstances();
		VoxelMedHism->ClearInstances();
		VoxelHism->ClearInstances();
		FlushPersistentDebugLines(GetWorld());
	}
	AnalysisDone = false;

	// Note: Sun is rendered via blueprint with a 90-deg north offset to match these coordinates.
	RenderHeliostats(OutResult, HelioVerticalOffset);

	UE_LOG(LogTemp, Warning, TEXT("NUM VOXELS : %d"), Field.NumVoxels);

	for (size_t i = 0; i < Field.NumVoxels; i++)
	{
		float irrad = OutResult.Irrads[i];
		TotalIrrad += irrad;
		float3 voxelLoc = GetVoxelLocFromIndex(i, VoxelSize, Field.Radius, Field.MinHeight);

		if (irrad >= IrradHighThreshold)
		{
			auto [x, y, z] = voxelLoc;
			FVector loc = FVector(-x, y, z);
			TotalThresIrrad += irrad;

			// undo mesh size scaling so position unaffected
			loc *= M_TO_CM / VoxelSize;
			VoxelHism->AddInstance(FTransform(loc));
			NumIrradVoxels++;
		}
		else if (irrad >= IrradMedThreshold)
		{
			auto [x, y, z] = voxelLoc;
			FVector loc = FVector(-x, y, z);
			TotalThresIrrad += irrad;

			// undo mesh size scaling so position unaffected
			loc *= M_TO_CM / VoxelSize;
			VoxelMedHism->AddInstance(FTransform(loc));
			NumIrradVoxels++;
		}
	}
	UE_LOG(LogTemp, Warning, TEXT("Threshold voxels: %d"), NumIrradVoxels);
	UE_LOG(LogTemp, Warning, TEXT("Total irrad: %.0f. Total irrad > threshold: %.0f"), TotalIrrad, TotalThresIrrad);

	AnalysisDone = true;

	return true;
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
		VoxelMedHism->ClearInstances();
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

	// --- Render voxels with adaptive LOD
	// Detail zone (near aim point): render individual voxels at native size
	// Far field: merge MergeFactor^3 voxel blocks into single larger instances
	TotalIrrad = 0.0f;
	TotalThresIrrad = 0.0f;
	NumIrradVoxels = 0;
	NumThresholdVoxels = 0;

	const FVector AimENU(AimStrat.Params.X, AimStrat.Params.Y, AimStrat.Params.Z);
	const float DetailRadiusSq = DetailZoneRadius * DetailZoneRadius;
	const int Vs = Field.VoxelSize;
	const int VoxelsPerSide = (Field.Radius * 2) / Vs;
	const int VoxelsPerPlane = VoxelsPerSide * VoxelsPerSide;
	const int NumVoxelsZ = (Field.MaxHeight - Field.MinHeight) / Vs;
	const int Mf = FMath::Clamp(MergeFactor, 2, 8);
	const int MergedVoxelSize = Vs * Mf;

	// Build a threshold bitmap so we can check block occupancy for merging
	TArray<uint8> AboveThreshold;
	AboveThreshold.SetNumZeroed(Field.NumVoxels);

	// First pass: accumulate totals and mark above-threshold voxels
	for (int vi = 0; vi < Field.NumVoxels; ++vi)
	{
		const float Irrad = OutResult.Irrads[vi];
		if (Irrad > 0.0f) ++NumIrradVoxels;
		TotalIrrad += Irrad;

		if (Irrad >= IrradMedThreshold)
		{
			TotalThresIrrad += Irrad;
			++NumThresholdVoxels;
			AboveThreshold[vi] = Irrad >= IrradHighThreshold ? 2 : 1;
		}
	}

	// Track which voxels have been handled by a merged block
	TArray<uint8> Handled;
	Handled.SetNumZeroed(Field.NumVoxels);

	int32 HighRendered = 0;
	int32 MedRendered = 0;
	int32 MergedRendered = 0;
	int32 DetailRendered = 0;
	int32 TotalRendered = 0;

	// Second pass: collect detail zone voxels, sort by irradiance so hotspot renders first
	struct FDetailVoxel { int Index; float Irrad; float3 Loc; };
	TArray<FDetailVoxel> DetailVoxels;

	for (int vi = 0; vi < Field.NumVoxels; ++vi)
	{
		if (!AboveThreshold[vi]) continue;

		float3 VoxLoc = GetVoxelLocFromIndex(vi, Vs, Field.Radius, Field.MinHeight);

		const float Dx = VoxLoc.x - AimENU.X;
		const float Dy = VoxLoc.y - AimENU.Y;
		const float Dz = VoxLoc.z - AimENU.Z;
		if ((Dx*Dx + Dy*Dy + Dz*Dz) > DetailRadiusSq) continue;

		DetailVoxels.Add({vi, OutResult.Irrads[vi], VoxLoc});
	}

	// Highest irradiance first — ensures the core hotspot always renders
	DetailVoxels.Sort([](const FDetailVoxel& A, const FDetailVoxel& B) { return A.Irrad > B.Irrad; });

	// Detail gets up to 75% of the budget; merged fills the rest
	const int32 DetailBudget = MaxVoxelInstances * 3 / 4;

	for (const auto& Dv : DetailVoxels)
	{
		if (DetailRendered >= DetailBudget) break;

		Handled[Dv.Index] = 1;

		float Ux = Dv.Loc.x * M_TO_CM * -1.f;
		float Uy = Dv.Loc.y * M_TO_CM;
		float Uz = Dv.Loc.z * M_TO_CM;
		FTransform Trans(FRotator(0,0,0), FVector(Ux, Uy, Uz), FVector(Vs, Vs, Vs));

		if (AboveThreshold[Dv.Index] == 2)
		{
			VoxelHism->AddInstance(Trans, true);
			++HighRendered;
		}
		else
		{
			VoxelMedHism->AddInstance(Trans, true);
			++MedRendered;
		}
		++DetailRendered;
		++TotalRendered;
	}

	// Third pass: fill remaining budget with merged blocks outside the detail zone
	for (int gz = 0; gz + Mf <= NumVoxelsZ && TotalRendered < MaxVoxelInstances; gz += Mf)
	{
		for (int gy = 0; gy + Mf <= VoxelsPerSide && TotalRendered < MaxVoxelInstances; gy += Mf)
		{
			for (int gx = 0; gx + Mf <= VoxelsPerSide && TotalRendered < MaxVoxelInstances; gx += Mf)
			{
				// Check if any voxel in this block is above threshold and not already handled
				bool bAnyAbove = false;
				float MaxIrrad = 0.0f;
				for (int dz = 0; dz < Mf; ++dz)
					for (int dy = 0; dy < Mf; ++dy)
						for (int dx = 0; dx < Mf; ++dx)
						{
							const int vi = (gx+dx) + (gy+dy)*VoxelsPerSide + (gz+dz)*VoxelsPerPlane;
							if (AboveThreshold[vi] && !Handled[vi])
							{
								bAnyAbove = true;
								MaxIrrad = FMath::Max(MaxIrrad, OutResult.Irrads[vi]);
							}
						}

				if (!bAnyAbove) continue;

				// Mark all voxels in block as handled
				for (int dz = 0; dz < Mf; ++dz)
					for (int dy = 0; dy < Mf; ++dy)
						for (int dx = 0; dx < Mf; ++dx)
							Handled[(gx+dx) + (gy+dy)*VoxelsPerSide + (gz+dz)*VoxelsPerPlane] = 1;

				// Render merged block — corner position, scaled up
				const float WorldX = static_cast<float>(gx) * Vs - Field.Radius;
				const float WorldY = static_cast<float>(gy) * Vs - Field.Radius;
				const float WorldZ = static_cast<float>(gz) * Vs + Field.MinHeight;

				FTransform Trans(FRotator(0,0,0),
					FVector(WorldX * M_TO_CM * -1.f, WorldY * M_TO_CM, WorldZ * M_TO_CM),
					FVector(MergedVoxelSize, MergedVoxelSize, MergedVoxelSize));

				if (MaxIrrad >= IrradHighThreshold)
				{
					VoxelHism->AddInstance(Trans, true);
					++HighRendered;
				}
				else
				{
					VoxelMedHism->AddInstance(Trans, true);
					++MedRendered;
				}
				++MergedRendered;
				++TotalRendered;
			}
		}
	}

	UE_LOG(LogFlux, Warning, TEXT("Total: %.2f kW; voxels with nonzero irrad: %d"), TotalIrrad, NumIrradVoxels);
	UE_LOG(LogFlux, Warning, TEXT("Flux Threshold 1: %.2f kW/m2 ; Flux Threshold 2: %.2f kW/m2"), IrradMedThreshold, IrradHighThreshold);
	UE_LOG(LogFlux, Warning, TEXT("Total irrad above threshold: %.2f kW (%d voxels)"), TotalThresIrrad, NumThresholdVoxels);
	UE_LOG(LogFlux, Warning, TEXT("Rendered: %d high + %d med = %d instances (%d detail, %d merged, max: %d)"),
		HighRendered, MedRendered, TotalRendered, DetailRendered, MergedRendered, MaxVoxelInstances);

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

	UE_LOG(LogFlux, Warning, TEXT("%d heliostats rendered"), Field.NumHelios);
}


//
// --- Setters & Update Utilities
//
void AFluxManager::SetSite(const int Option)
{
	UE_LOG(LogTemp, Warning, TEXT("New site %d selected. Overwriting other inputs."), Option);

	const auto Mode = UIrradianceFunctionLibrary::GetGameMode(this);
	SelectedSiteConfig = Mode->GetSiteConfig(Option);
	CspSite = SelectedSiteConfig.SiteType;
	AimStrat = SelectedSiteConfig.AimStrategy;
	Datetime = SelectedSiteConfig.DateTime;
	Field = SelectedSiteConfig.Field;
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
                          const float AimRingParam1, const float AimRingParam2,
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
		AimStrat.AimStrategyType = EAimStrategyType::Aim_Ring;
		AimStrat.Params.X = AimRingParam1;
		AimStrat.Params.Y = AimRingParam2;
		AimStrat.Params.Z = 0;
		break;
	default:
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
