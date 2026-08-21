// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include "CoreMinimal.h"
#include "LogStream.h"
#include "SiteConfigTypes.h"
#include "GameFramework/Actor.h"
#include "SunPosition.h"
#include "FluxManager.generated.h"

// NB: the *Hism member names below are historical -- these are plain ISMs since the
// switch away from UHierarchicalInstancedStaticMeshComponent. The names are kept
// because they are UPROPERTYs bound by BP_FluxManager.
class UInstancedStaticMeshComponent;
class UStaticMesh;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnProcessingCompleteDelegate, bool, bSuccess);

UCLASS()
class FFIAM_API AFluxManager : public AActor
{
	GENERATED_BODY()

public:
	AFluxManager();
	virtual ~AFluxManager() override;

protected:
	virtual void BeginPlay() override;

	// Generic tower
	UPROPERTY(EditAnywhere, Category = "Field")
	TObjectPtr<AActor> TallTower;

	// NSTTF tower
	UPROPERTY(EditAnywhere, Category = "Field")
	TObjectPtr<AActor> NsttfTower;

public:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Components")
	UStaticMeshComponent* RootMesh;
	
	UPROPERTY(EditAnywhere, Category = "Voxel")
	UInstancedStaticMeshComponent* VoxelHism ;
	
	UPROPERTY(EditAnywhere, Category = "Voxel")
	UStaticMesh* VoxelMesh;

	UPROPERTY(EditAnywhere, Category = "Heliostats")
	UInstancedStaticMeshComponent* HelioHism ;
	
	UPROPERTY(EditAnywhere, Category = "Voxel")
	UStaticMesh* HelioMesh;
	
	UPROPERTY(EditAnywhere, Category = "Field")
	UStaticMeshComponent* AimPointComponent ;
	
	UPROPERTY(EditAnywhere, Category = "Field")
	float HelioSide = 90.0f;
	
	UPROPERTY(BlueprintAssignable, Category = "Analysis")
	FOnProcessingCompleteDelegate OnProcessingComplete;  // triggers result update
	
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void DoAnalysisAsync();

	UFUNCTION()
	void OnAnalysisCompleted(bool bSuccess);
	
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void SetSite(int Option);

	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void UpdateParameters(int SiteIdx, int NewYear, int NewMonth, int NewDay, int NewHour);

	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void SetDatetime(int NewYear, int NewMonth, int NewDay, int NewHour);
	
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void SetAim(int AimIdx,
	            float AimPtParam1, float AimPtParam2, float AimPtParam3,
	            float AimRingParam1, float AimRingParam2, float AimRingParam3,
	            float AimVecParam1, float AimVecParam2, float AimVecParam3);
	
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void SetThresholds(float Thres1, float Thres2);

	// Colour-ramp scaling. Relative (default) maps the ramp to this run's own peak,
	// which maximises contrast but makes runs incomparable. Absolute pins the top of
	// the ramp to PeakKwM2 so colours mean the same thing across sites, times of day
	// and aim strategies -- use it whenever two runs are shown side by side.
	// Re-colours the existing voxels in place; no re-analysis needed.
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void SetColorScale(bool bAbsolute, float PeakKwM2);

	// Top of the ramp actually in use (kW/m2) -- drive the scale legend from this.
	UFUNCTION(BlueprintPure, Category = "Analysis")
	float GetColorScalePeak() const;

	// Fraction [0,1] along the colour ramp that a given irradiance maps to, using the
	// SAME log mapping the voxels do -- so the legend's Threshold1/Threshold2 tick marks
	// cannot drift from the voxel colours. Drive a tick's position from
	// RampInputFor(Threshold2); Threshold1 sits at 0 and GetColorScalePeak() at 1 by
	// construction. Position only -- the material's 0.15 opacity/colour floor does not
	// enter here, so it does not need replicating in the legend.
	UFUNCTION(BlueprintPure, Category = "Analysis")
	float RampInputFor(float IrradKwM2) const;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Voxel")
	bool bAbsoluteColorScale = false;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Voxel", meta = (ClampMin = "0.01"))
	float ColorScalePeak = 1000.0f;

	// Relative mode only: which percentile of the drawn irradiances becomes the top of
	// the colour ramp. Flux is long-tailed, so normalising to the true maximum spends
	// most of the ramp on a handful of focal-spot voxels and leaves the bulk of the
	// cloud indistinguishable. Lower this to spread more colour across the bulk;
	// raise it toward 100 to recover the old max-normalised behaviour.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Voxel",
	          meta = (ClampMin = "50.0", ClampMax = "100.0"))
	float ColorScalePercentile = 99.0f;

	LogStream Stream;

	bool AnalysisDone;
	
	// Max instances rendered before voxels get dropped (hottest kept). Caps GPU cost on
	// large sites. At 1 km / 1 m the shell is ~16M voxels, so the old 4M default dropped
	// most of the faint outer shell and shrank the silhouette; raise toward the shell
	// count to restore it (watch translucent overdraw / framerate). Editor-tunable.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Voxel",
	          meta = (ClampMin = "100000", UIMin = "1000000", UIMax = "32000000"))
	int32 MaxVoxelInstances = 12000000;

	// Greedy-meshing merge tolerance: adjacent shell voxels merge into one stretched box
	// (XY rectangles, per z-slice) only while sharing an irradiance band, so a merged box
	// is coloured by its hottest voxel to within MergeIrradRatio. 1.0 = merge only
	// equal-irradiance voxels (least colour flattening, least reduction); higher = more
	// merging and fewer instances at the cost of colour detail across a rectangle.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Voxel",
	          meta = (ClampMin = "1.0", UIMin = "1.0", UIMax = "4.0"))
	float MergeIrradRatio = 1.5f;

	// Last-run voxel render stats, for a HUD readout ("showing X of Y"). Set in DoAnalysis.
	UPROPERTY(BlueprintReadOnly, Category = "Voxel")
	int32 LastShellVoxelCount = 0;

	UPROPERTY(BlueprintReadOnly, Category = "Voxel")
	int32 LastDrawnVoxelCount = 0;

	// Do NOT delete this! Need early reference, otherwise BP_SunPosition blueprint will break on UE launch.
	FSunPositionData TestSunData;
	
	UPROPERTY(editanywhere, BlueprintReadWrite, Category = "Analysis")
	FSimpleField Field;

	UPROPERTY(editanywhere, BlueprintReadWrite, Category = "Analysis")
	FDateInfo Datetime;

	UPROPERTY(editanywhere, BlueprintReadWrite, Category = "Analysis")
	ECspSite CspSite;
	
	UPROPERTY(editanywhere, BlueprintReadWrite, Category = "Analysis")
	FAimStrategy AimStrat;
	
	UPROPERTY(BlueprintReadWrite, Category = "Site")
	float Lat;
	
	UPROPERTY(BlueprintReadWrite, Category = "Site")
	float Lng;
	
	UPROPERTY(BlueprintReadWrite, Category = "Site")
	float Timezone;
	
	UPROPERTY(BlueprintReadWrite, Category = "Datetime")
	int Year;
	
	UPROPERTY(BlueprintReadWrite, Category = "Datetime")
	int Month;
	
	UPROPERTY(BlueprintReadWrite, Category = "Datetime")
	int Day;
	
	UPROPERTY(BlueprintReadWrite, Category = "Datetime")
	int Hour;
	
	UPROPERTY(BlueprintReadWrite, Category = "Site")
	FVector3f AimParams;
	
	UPROPERTY(BlueprintReadWrite, Category = "Analysis")
	float Threshold1;
	
	UPROPERTY(BlueprintReadWrite, Category = "Analysis")
	float Threshold2;
	
	UPROPERTY(BlueprintReadOnly, Category = "Analysis")
	float TotalIrrad;
	
	UPROPERTY(BlueprintReadOnly, Category = "Analysis")
	float TotalThresIrrad;

	// Number of voxels receiving some irradiance
	UPROPERTY(BlueprintReadOnly, Category = "Analysis")
	int NumIrradVoxels;

	// Number of voxels receiving irradiance above threshold
	UPROPERTY(BlueprintReadOnly, Category = "Analysis")
	int NumThresholdVoxels;

	UPROPERTY(BlueprintReadOnly, Category = "Analysis")
	bool bIsAnalyzing = false;

private:
	FSiteConfig SelectedSiteConfig;

	void RenderHeliostats(const FAnalysisResult& Result, float HelioVerticalOffset);

	// Recomputes every instance's ramp input from the stored irradiances.
	void ApplyColorScale();

	// The ONE definition of the flux -> ramp-input (T) mapping, split so it is computed
	// per-run once and applied cheaply per instance. GetRampLogRange picks the peak for
	// the current scale mode and returns the log range; ApplyRampLogRange maps one value
	// through it. ApplyColorScale (4M calls) and RampInputFor (the legend, ~1 call) both
	// go through these, so the tick marks and the voxels can never disagree.
	void GetRampLogRange(float& OutLogLo, float& OutInvLogRange, float& OutPeakKwM2) const;
	static float ApplyRampLogRange(float IrradKwM2, float LogLo, float InvLogRange);

	// Irradiance per rendered instance, indexed by instance index, so the colour
	// scale can be re-applied without re-running the analysis. One array since the
	// high/med split collapsed (TODO item 5): every occupied-shell voxel renders into
	// VoxelHism and is distinguished only by ramp colour.
	TArray<float> VoxelInstanceIrrads;

	// Top of the ramp in relative mode: the ColorScalePercentile-th percentile of the
	// last run's DRAWN irradiances, not their maximum. See ApplyColorScale.
	float LastShellRampTop = 1.0f;

	// NB: voxel index <-> world position and the grid dimensions come from the
	// library (core/voxel.h). The indexing convention is the contract between the
	// CUDA kernel that writes Irrads and this renderer that reads it, so it must
	// have exactly one definition -- a second copy would misplace every voxel on a
	// grid-layout change with no compile error.

};
