// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include "CoreMinimal.h"
#include "FFIAM.h"
#include "SiteConfigTypes.h"
#include "LogStream.h"
#include "memory/pool.h"
#include "ffiam/ffiam.h"
#include "GameFramework/GameMode.h"
#include "Engine/World.h"
#include "HAL/Runnable.h"
#include "HAL/RunnableThread.h"
#include "IrradianceMode.generated.h"


class FAnalysisTask;
class UUserWidget;
class IInputProcessor;
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnSitesRefreshedDelegate, const TArray<FSiteConfig>&, SiteConfigs);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAnalysisCompleteDelegate, bool, bSuccess);
DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnAnalysisProgressDelegate, float, Progress);


/**
 * Houses analysis integration and memory management since other objects can access and call into this.
 * Keep integration uni-directional: objects call into this Mode; it doesn't call out.
 * Keep all ffiam.h use (structs, etc.) contained within this class.
 */
UCLASS()
class FFIAM_API AIrradianceMode : public AGameMode
{
	GENERATED_BODY()

public:
	AIrradianceMode();
	virtual ~AIrradianceMode() override;
	
	LogStream Stream;

	/**
	 * Start analysis using specified site parameters.
	 * @param CspSiteType
	 * @param Field
	 * @param HeliostatDesign
	 * @param AimStrategy
	 * @param Datetime
	 * @param OutResult
	 * @return
	 */
	bool DoAnalysis(const ECspSite CspSiteType,
	                const FSimpleField& Field,
	                const FHeliostatDesign& HeliostatDesign,
	                const FAimStrategy& AimStrategy,
	                const FDateInfo& Datetime,
	                FAnalysisResult& OutResult);

	/**
	 * Start asynchronous analysis using specified site parameters.
	 * Results are delivered via OnAnalysisComplete delegate.
	 */
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void DoAnalysisAsync(const ECspSite CspSiteType,
	                     const FSimpleField& Field,
	                     const FHeliostatDesign& HeliostatDesign,
	                     const FAimStrategy& AimStrategy,
	                     const FDateInfo& Datetime);

	/**
	 * Start analysis using un-modified site data from JSON file.
	 * @param SiteIndex
	 * @param OutResult
	 * @return
	 */
	bool DoAnalysisFromPresetSite(int SiteIndex, FAnalysisResult& OutResult);

	/**
	 * Start async analysis using un-modified site data from JSON file.
	 */
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void DoAnalysisFromPresetSiteAsync(int SiteIndex);
	
	UFUNCTION(blueprintcallable, Category = "Analysis")
	void LoadSiteConfigFromJson();

	UPROPERTY(BlueprintAssignable, Category = "Analysis")
	FOnSitesRefreshedDelegate OnSiteConfigsRefreshed;

	UPROPERTY(BlueprintAssignable, Category = "Analysis")
	FOnAnalysisCompleteDelegate OnAnalysisComplete;

	UPROPERTY(BlueprintAssignable, Category = "Analysis")
	FOnAnalysisProgressDelegate OnAnalysisProgress;
	
	UFUNCTION(BlueprintCallable)
	TArray<FSiteConfig> GetLoadedSiteConfigs() { return LoadedSiteConfigs;}

	/**
	 * Get the last analysis result. Only valid after OnAnalysisComplete fires with success.
	 */
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	FAnalysisResult GetLastAnalysisResult() const { return LastAnalysisResult; }

	/**
	 * Check if an analysis is currently running.
	 */
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	bool IsAnalysisRunning() const { return bAnalysisRunning; }

	UFUNCTION()
	FSiteConfig GetSiteConfig(int SiteIdx);

	/**
	 * Toggle visibility of all on-screen HUD widgets (overlay + stats readout).
	 * Bound to the 'H' key via a Slate input pre-processor; lets you hide the
	 * HUD for clean screenshots and restores each widget's prior visibility.
	 */
	void ToggleHud();

protected:
	friend FAnalysisTask;

	// Load site configs here (runs before any actor BeginPlay) so the FluxManager's
	// BeginPlay -> SetSite(0) finds them ready instead of an empty list.
	virtual void InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage) override;
	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;


	UFUNCTION()
	void ReinitializePool();

	UFUNCTION()
	// Returns false if the voxel grid will not fit the arena; caller must abort.
	bool PrepMemoryForAnalysis();

	// Field data
	void* MemoryArena;
	Pool MemoryPool;

	// Voxel data
	void* VoxelArena;
	Pool VoxelPool;

	/**
	 * Location of JSON site configuration files relative to project config directory.
	 * Must ensure this directory is packaged by adding an entry to "Additional Non-Asset Directories to copy" in Project Settings -> Packaging
	 * E.g., "../Config/FFIAMData/SiteConfigs"
	 */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Analysis")
	FString JsonSiteConfigRelativeDir = "FFIAMData/SiteConfigs";
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Analysis")
	FString SiteCsvRelativeDir = "FFIAMData/FieldData";
	
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Analysis")
	TArray<FSiteConfig> LoadedSiteConfigs;
	

private:
	const size_t NumFieldMemory = GB(2);
	const size_t FieldChunkSize = MB(256);
	
	// Irrads is the voxel pool's only allocation, so the arena is a single chunk
	// (pool_alloc hands back the LAST chunk -- splitting the arena would put
	// Irrads at a non-zero offset for no benefit). Must hold the largest grid we
	// support: 1 km radial at 1 m voxels = 788M voxels = 2.94 GiB.
	const size_t NumVoxelMemory = MB(4096);
	const size_t VoxelChunkSize = NumVoxelMemory;

	bool bPoolFirstInitialized = false;

	// Store analysis structs and arrays to easily reset between runs
	field_layout NativeField = {};
	heliostat_design NativeHelioDesign = {};
	date_info NativeDatetime = {};
	csp_site NativeCspSite = {};
	aim_strategy NativeAimStrategy = {};
	float* Irrads = nullptr;
	float* MiscData = nullptr;
    float3* Misc3Data = nullptr;

	// Async analysis support
	FAnalysisResult LastAnalysisResult;
	bool bAnalysisRunning = false;
	class FAnalysisTask* CurrentAnalysisTask = nullptr;
	FRunnableThread* AnalysisThread = nullptr;

	// Thread-safe analysis execution
	bool DoAnalysisInternal(const ECspSite CspSiteType,
	                        const FSimpleField& Field,
	                        const FHeliostatDesign& HeliostatDesign,
	                        const FAimStrategy& AimStrategy,
	                        const FDateInfo& Datetime,
	                        FAnalysisResult& OutResult);

	void CleanupAnalysisTask();

	// HUD-toggle state. SavedHudVisibility remembers each widget's visibility
	// while hidden so ToggleHud() can restore it exactly (important: the overlay
	// may be SelfHitTestInvisible so camera drags pass through it).
	bool bHudHidden = false;
	TMap<UUserWidget*, uint8> SavedHudVisibility;
	TSharedPtr<IInputProcessor> HudToggleProcessor;

	// Select site 0 in the HUD's site combo once it has populated, so it doesn't
	// show a blank selection at startup. Retried on a short timer because the
	// widget populates asynchronously (from the OnSiteConfigsRefreshed broadcast).
	void SelectDefaultSiteInHud();
	FTimerHandle HudDefaultSiteTimer;
	int32 HudDefaultSiteTries = 0;
};


