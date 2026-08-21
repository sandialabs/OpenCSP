// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#include "IrradianceMode.h"

#include "AnalysisTask.h"
#include "ffiam/ffiam.h"
#include "FFIAM.h"
#include "IrradianceFunctionLibrary.h"
#include "IrradianceGameState.h"
#include "LogChannels.h"
#include "HAL/PlatformFileManager.h" // For FPlatformFileManager
#include "Misc/Paths.h"             // For FPaths
#include "SiteConfigTypes.h"
#include "Async/TaskGraphInterfaces.h" // For FFunctionGraphTask
#include "Blueprint/UserWidget.h"
#include "Blueprint/WidgetBlueprintLibrary.h"
#include "Blueprint/WidgetTree.h"
#include "Components/ComboBoxString.h"
#include "Framework/Application/SlateApplication.h"
#include "Framework/Application/IInputProcessor.h"
#include "Widgets/SWidget.h"
#include "InputCoreTypes.h"
#include "TimerManager.h"


// Global Slate input pre-processor that toggles the HUD on a hotkey. Using a
// pre-processor (rather than a PlayerController binding) makes the key work
// regardless of the game's input mode / UI focus — but we deliberately ignore
// it while an editable text field is focused, so numeric HUD entry isn't eaten.
namespace
{
	class FHudToggleInputProcessor : public IInputProcessor
	{
	public:
		TWeakObjectPtr<AIrradianceMode> Mode;
		FKey ToggleKey = EKeys::H;

		virtual void Tick(const float /*DeltaTime*/, FSlateApplication& /*SlateApp*/,
		                  TSharedRef<ICursor> /*Cursor*/) override {}

		virtual bool HandleKeyDownEvent(FSlateApplication& SlateApp,
		                                const FKeyEvent& InKeyEvent) override
		{
			if (InKeyEvent.GetKey() != ToggleKey || InKeyEvent.IsRepeat())
			{
				return false;
			}

			// Don't steal the key while the user is typing into a HUD text box.
			const TSharedPtr<SWidget> Focused = SlateApp.GetKeyboardFocusedWidget();
			if (Focused.IsValid() && Focused->GetType().ToString().Contains(TEXT("EditableText")))
			{
				return false;
			}

			if (AIrradianceMode* M = Mode.Get())
			{
				M->ToggleHud();
				return true; // consume the key
			}
			return false;
		}

		virtual const TCHAR* GetDebugName() const override { return TEXT("FfiamHudToggle"); }
	};
}


AIrradianceMode::AIrradianceMode()
{
	MemoryArena = FMemory::Malloc(NumFieldMemory);
	FMemory::Memzero(MemoryArena, NumFieldMemory);
	pool_init(&MemoryPool, MemoryArena, NumFieldMemory, FieldChunkSize, DEFAULT_ALIGNMENT);
	
	VoxelArena = FMemory::Malloc(NumVoxelMemory);
	FMemory::Memzero(VoxelArena, NumVoxelMemory);
	pool_init(&VoxelPool, VoxelArena, NumVoxelMemory, VoxelChunkSize, DEFAULT_ALIGNMENT);
	
	bPoolFirstInitialized = true;
	
}


AIrradianceMode::~AIrradianceMode()
{
	CleanupAnalysisTask();

	pool_free_all(&MemoryPool);
	FMemory::Free(MemoryArena);

	pool_free_all(&VoxelPool);
	FMemory::Free(VoxelArena);
}


void AIrradianceMode::InitGame(const FString& MapName, const FString& Options, FString& ErrorMessage)
{
	Super::InitGame(MapName, Options, ErrorMessage);

	// InitGame runs before any actor's BeginPlay, so loading here guarantees the
	// site configs exist when AFluxManager::BeginPlay calls SetSite(0). BeginPlay
	// below still (re)loads + broadcasts OnSiteConfigsRefreshed so the HUD widget,
	// which subscribes later, still gets its dropdown populated.
	LoadSiteConfigFromJson();
}


void AIrradianceMode::BeginPlay()
{
	Super::BeginPlay();

	AIrradianceGameState* State = GetGameState<AIrradianceGameState>();
	checkf(State, TEXT("GameState not found"));
	
	UE_LOG(LogFlux, Warning, TEXT("Calling TestPrint to verify DLL loaded."));
	
	auto oldCout = std::cout.rdbuf(&Stream);
	std::cout << "Redirecting stdout for Test." << std::endl;
	cudaError_t TestStatus = cudaErrorUnknown;
	try
	{
		TestStatus = TestPrint(1);
	} catch (...) {
		std::cerr << "Failed TestPrint. Unknown exception caught!" << std::endl;
	}
	std::cout.rdbuf(oldCout);
	
	if (TestStatus != cudaSuccess)
	{
		UE_LOG(LogFlux, Error, TEXT("Library test call failed!\n"));
	}
	else
	{
		UE_LOG(LogFlux, Warning, TEXT("Library test call succeeded!"));
	}

	LoadSiteConfigFromJson();

	State->LoadedSiteConfigs = LoadedSiteConfigs;

	// Give the HUD's site combo a default selection once it has populated.
	HudDefaultSiteTries = 0;
	GetWorldTimerManager().SetTimer(HudDefaultSiteTimer, this,
		&AIrradianceMode::SelectDefaultSiteInHud, 0.1f, true, 0.1f);

	// Register the HUD-toggle hotkey (H). See FHudToggleInputProcessor above.
	if (FSlateApplication::IsInitialized())
	{
		TSharedRef<FHudToggleInputProcessor> Processor = MakeShared<FHudToggleInputProcessor>();
		Processor->Mode = this;
		FSlateApplication::Get().RegisterInputPreProcessor(Processor);
		HudToggleProcessor = Processor;
		UE_LOG(LogFlux, Warning, TEXT("HUD-toggle hotkey registered (press H to hide/show the HUD)."));
	}
}


void AIrradianceMode::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	if (HudToggleProcessor.IsValid() && FSlateApplication::IsInitialized())
	{
		FSlateApplication::Get().UnregisterInputPreProcessor(HudToggleProcessor.ToSharedRef());
	}
	HudToggleProcessor.Reset();

	GetWorldTimerManager().ClearTimer(HudDefaultSiteTimer);

	Super::EndPlay(EndPlayReason);
}


void AIrradianceMode::SelectDefaultSiteInHud()
{
	HudDefaultSiteTries++;

	const int32 NumSites = LoadedSiteConfigs.Num();
	if (NumSites > 0)
	{
		TArray<UUserWidget*> Widgets;
		UWidgetBlueprintLibrary::GetAllWidgetsOfClass(this, Widgets, UUserWidget::StaticClass(), false);
		for (UUserWidget* W : Widgets)
		{
			if (!IsValid(W)) continue;

			// The site combo is "siteSelector"; fall back to the ComboBoxString
			// whose option count matches the loaded site list (tells it apart from
			// the aim-strategy combo).
			UComboBoxString* Combo = Cast<UComboBoxString>(W->GetWidgetFromName(TEXT("siteSelector")));
			if (!Combo && W->WidgetTree)
			{
				TArray<UWidget*> All;
				W->WidgetTree->GetAllWidgets(All);
				for (UWidget* Wid : All)
				{
					UComboBoxString* C = Cast<UComboBoxString>(Wid);
					if (C && C->GetOptionCount() == NumSites) { Combo = C; break; }
				}
			}

			if (Combo && Combo->GetOptionCount() > 0)
			{
				// SetSelectedIndex fires OnSelectionChanged -> SetSite(0), syncing
				// the display with the FluxManager's default site.
				if (Combo->GetSelectedIndex() < 0)
				{
					Combo->SetSelectedIndex(0);
				}
				GetWorldTimerManager().ClearTimer(HudDefaultSiteTimer);
				return;
			}
		}
	}

	if (HudDefaultSiteTries >= 50)  // ~5s; give up rather than poll forever
	{
		GetWorldTimerManager().ClearTimer(HudDefaultSiteTimer);
	}
}


void AIrradianceMode::ToggleHud()
{
	TArray<UUserWidget*> Widgets;
	UWidgetBlueprintLibrary::GetAllWidgetsOfClass(this, Widgets, UUserWidget::StaticClass(), false);

	bHudHidden = !bHudHidden;
	if (bHudHidden)
	{
		// Remember each widget's current visibility, then collapse it.
		SavedHudVisibility.Reset();
		for (UUserWidget* W : Widgets)
		{
			if (!IsValid(W)) continue;
			SavedHudVisibility.Add(W, static_cast<uint8>(W->GetVisibility()));
			W->SetVisibility(ESlateVisibility::Collapsed);
		}
	}
	else
	{
		// Restore each widget's remembered visibility.
		for (UUserWidget* W : Widgets)
		{
			if (!IsValid(W)) continue;
			const uint8* Saved = SavedHudVisibility.Find(W);
			W->SetVisibility(Saved ? static_cast<ESlateVisibility>(*Saved) : ESlateVisibility::Visible);
		}
		SavedHudVisibility.Reset();
	}

	UE_LOG(LogFlux, Warning, TEXT("HUD %s (H)"), bHudHidden ? TEXT("hidden") : TEXT("shown"));
}


void AIrradianceMode::LoadSiteConfigFromJson()
{
	// Load sites from JSON files. Replaces existing ones.
	int NumLoaded = UIrradianceFunctionLibrary::LoadAllSiteConfigsFromDirectory(JsonSiteConfigRelativeDir, LoadedSiteConfigs);
	if (NumLoaded > 0)
	{
		OnSiteConfigsRefreshed.Broadcast(LoadedSiteConfigs);
	}
}


void AIrradianceMode::ReinitializePool()
{
	if (!bPoolFirstInitialized) return;

	pool_free_all(&MemoryPool);
	FMemory::Memzero(MemoryArena, NumFieldMemory);
	pool_init(&MemoryPool, MemoryArena, NumFieldMemory, FieldChunkSize, DEFAULT_ALIGNMENT);
	
	pool_free_all(&VoxelPool);
	FMemory::Memzero(VoxelArena, NumVoxelMemory);
	pool_init(&VoxelPool, VoxelArena, NumVoxelMemory, VoxelChunkSize, DEFAULT_ALIGNMENT);
}


FSiteConfig AIrradianceMode::GetSiteConfig(const int SiteIdx)
{
	if (SiteIdx < 0 || SiteIdx >= LoadedSiteConfigs.Num())
	{
		UE_LOG(LogFlux, Error, TEXT("Invalid site index %d"), SiteIdx);
		return {};
	}
	
	return LoadedSiteConfigs[SiteIdx];
}


bool AIrradianceMode::PrepMemoryForAnalysis()
{
	// The library writes nVoxels floats into the Irrads chunk with no bounds
	// check of its own, so a grid that outgrows the arena corrupts the heap and
	// dies inside FieldAnalysis with a bare SIGSEGV. Refuse the run instead.
	const size_t IrradsBytes = static_cast<size_t>(NativeField.nVoxels) * sizeof(float);
	if (IrradsBytes > VoxelChunkSize)
	{
		UE_LOG(LogFlux, Error,
			TEXT("Voxel grid too large: %d voxels need %.2f GiB but the voxel arena chunk is %.2f GiB. ")
			TEXT("Raise NumVoxelMemory/VoxelChunkSize in IrradianceMode.h, increase VoxelSize, ")
			TEXT("or reduce FieldRadius / MaxHeight. (If you just changed those constants, rebuild ")
			TEXT("the FFIAM module and restart the editor -- site JSON is read at runtime but ")
			TEXT("the pool sizes are compiled in.)"),
			NativeField.nVoxels,
			static_cast<double>(IrradsBytes) / (1024.0 * 1024.0 * 1024.0),
			static_cast<double>(VoxelChunkSize) / (1024.0 * 1024.0 * 1024.0));
		return false;
	}

	pool_free(&MemoryPool, NativeField.locs);
	pool_free(&MemoryPool, NativeField.aimVs);
	pool_free(&MemoryPool, NativeField.moveAngles);
	pool_free(&MemoryPool, NativeField.helios);
	pool_free(&MemoryPool, MiscData);
	pool_free(&MemoryPool, Misc3Data);
	
	pool_free(&VoxelPool, Irrads);
	
	NativeField.locs = static_cast<float3*>(pool_alloc(&MemoryPool));
	NativeField.aimVs = static_cast<float3*>(pool_alloc(&MemoryPool));
	NativeField.moveAngles = static_cast<float*>(pool_alloc(&MemoryPool));
	NativeField.helios = static_cast<heliostat*>(pool_alloc(&MemoryPool));
	MiscData = static_cast<float*>(pool_alloc(&MemoryPool));
	Misc3Data = static_cast<float3*>(pool_alloc(&MemoryPool));
	
	Irrads = static_cast<float*>(pool_alloc(&VoxelPool));

	if (!Irrads)
	{
		UE_LOG(LogFlux, Error, TEXT("Voxel pool exhausted -- pool_alloc returned null for Irrads."));
		return false;
	}

	return true;
}


bool AIrradianceMode::DoAnalysis(const ECspSite CspSiteType,
                                 const FSimpleField& Field,
                                 const FHeliostatDesign& HeliostatDesign,
                                 const FAimStrategy& AimStrategy,
                                 const FDateInfo& Datetime,
                                 FAnalysisResult& OutResult)
{
	return DoAnalysisInternal(CspSiteType, Field, HeliostatDesign, AimStrategy, Datetime, OutResult);
}

bool AIrradianceMode::DoAnalysisInternal(const ECspSite CspSiteType,
                                         const FSimpleField& Field,
                                         const FHeliostatDesign& HeliostatDesign,
                                         const FAimStrategy& AimStrategy,
                                         const FDateInfo& Datetime,
                                         FAnalysisResult& OutResult)
{
	// Convert from UE to native FFIAM types
	NativeHelioDesign.nFacets = HeliostatDesign.NumFacets;
	NativeHelioDesign.nRows = HeliostatDesign.NumRows;
	NativeHelioDesign.nCols = HeliostatDesign.NumCols;
	NativeHelioDesign.facetWidth = HeliostatDesign.FacetWidth;
	NativeHelioDesign.facetHeight = HeliostatDesign.FacetHeight;
	if (!HeliostatDesign.FacetFile.IsEmpty())
	{
		const auto ModifiedPath = FPaths::ProjectDir() / TEXT("Config/FFIAMData/FieldData") / HeliostatDesign.FacetFile;
		FCStringAnsi::Strncpy(NativeHelioDesign.facetFile, TCHAR_TO_UTF8(*ModifiedPath), sizeof(NativeHelioDesign.facetFile));
	}
	else
	{
		NativeHelioDesign.facetFile[0] = '\0';
	}

	NativeDatetime.year = Datetime.Year;
	NativeDatetime.month = Datetime.Month;
	NativeDatetime.day = Datetime.Day;
	NativeDatetime.hour = Datetime.Hour;

	NativeAimStrategy.type = static_cast<aim_strategy_type>(AimStrategy.AimStrategyType); // Direct cast if enum values align
	NativeAimStrategy.params = AimStrategy.Params; // Uses operator float3()
	if (!AimStrategy.ImportFile.IsEmpty())
	{
		const auto ModifiedPath = FPaths::ProjectDir() / TEXT("Config/FFIAMData/FieldData") / AimStrategy.ImportFile;
		FCStringAnsi::Strncpy(NativeAimStrategy.file, TCHAR_TO_UTF8(*ModifiedPath), sizeof(NativeAimStrategy.file));
	}
	else
	{
		NativeAimStrategy.file[0] = '\0';
	}
        
	NativeCspSite = static_cast<csp_site>(CspSiteType); // Direct cast

	NativeField.lat = Field.Latitude;
	NativeField.lng = Field.Longitude;
	NativeField.timezone = Field.Timezone;
	NativeField.radius = Field.Radius;
	NativeField.towerHeight = Field.TowerHeight;
	NativeField.nHelios = Field.NumHelios;
	NativeField.zMax = Field.MaxHeight;
	NativeField.zMin = Field.MinHeight;
	if (!Field.HelioCoordinateFile.IsEmpty())
	{
		const auto ModifiedPath = FPaths::ProjectDir() / TEXT("Config/FFIAMData/FieldData") / Field.HelioCoordinateFile;
		FCStringAnsi::Strncpy(NativeField.helioFile, TCHAR_TO_UTF8(*ModifiedPath), sizeof(NativeField.helioFile));
	}
	else
	{
		NativeField.helioFile[0] = '\0';
	}

	// Generation layout (used only when helioFile is empty): grid vs radial.
	NativeField.layout = Field.Layout.Equals(TEXT("radial"), ESearchCase::IgnoreCase)
		? layout_radial : layout_grid;

	NativeField.voxelArea = Field.VoxelArea;
	NativeField.voxelSize = Field.VoxelSize;
	NativeField.nVoxels = Field.NumVoxels;

	if (!PrepMemoryForAnalysis())
	{
		return false;
	}

	const auto OldCout = std::cout.rdbuf(&Stream);
	std::cout << "Redirecting stdout" << std::endl;
	
	cudaError_t CudaStatus = cudaErrorUnknown;
	try
	{
		const float Reflectivity = 0.9f;
		const float DNI = 0.1f;
		const float Beta = 0.0094f;
		const float Ambient = 0.0f;

		// UAS-calibrated params only validated for NSTTF (218 heliostats, 600m radius).
		// Larger sites use original defaults — UAS calibration overestimates at long distances.
		const bool bUseUasParams = (NativeCspSite == site_nsttf);
		const float MinAttenuation    = bUseUasParams ? 0.42f : 1.0f;
		const float IrradExponent     = bUseUasParams ? 1.7f  : 2.0f;
		const float NRaysPerFacet     = 12.0f;
		const float FluxCorrectionScale = 1.0f;
		const float PreFocalScale     = bUseUasParams ? 0.6f  : 1.0f;

		UE_LOG(LogFlux, Warning, TEXT("Model params: %s (minAtten=%.2f, irradExp=%.1f, preFocal=%.1f)"),
			bUseUasParams ? TEXT("UAS-calibrated") : TEXT("Original defaults"),
			MinAttenuation, IrradExponent, PreFocalScale);

		const bool LogToFile = false;
		const bool Verbose = true;

		CudaStatus = FieldAnalysis(&NativeField,
		                           &NativeHelioDesign,
		                           &NativeAimStrategy,
		                           &NativeDatetime,
		                           Irrads,
		                           MiscData,
		                           Misc3Data,
		                           Verbose,
		                           LogToFile,
		                           Reflectivity,
		                           DNI,
		                           Beta,
		                           Ambient,
		                           MinAttenuation,
		                           IrradExponent,
		                           NRaysPerFacet,
		                           FluxCorrectionScale,
		                           PreFocalScale);

		OutResult.Irrads = Irrads;
		OutResult.MiscData = MiscData;
		OutResult.Misc3Data = Misc3Data;
		OutResult.Helios = NativeField.helios;
		OutResult.Locs = NativeField.locs;
		OutResult.AimVs = NativeField.aimVs;
		OutResult.MoveAngles = NativeField.moveAngles;
		
		OutResult.CudaStatus = CudaStatus;
		
	} catch (...) {
		std::cerr << "Unknown exception caught!" << std::endl;
	}
	std::cout.rdbuf(OldCout);
	
	UE_LOG(LogFlux, Warning, TEXT("Library analysis call complete"));

	if (CudaStatus != cudaSuccess)
	{
		UE_LOG(LogFlux, Warning, TEXT("Library analysis failed!\n"));
		return false;
	}
	

	return true;
}


bool AIrradianceMode::DoAnalysisFromPresetSite(int SiteIndex, FAnalysisResult& OutResult)
{
	if (SiteIndex < 0 || SiteIndex >= LoadedSiteConfigs.Num())
	{
		UE_LOG(LogFlux, Error, TEXT("Invalid site index %d"), SiteIndex);
		return false;
	}
	
	const auto SiteConfig = LoadedSiteConfigs[SiteIndex];
	const auto CspSiteType = SiteConfig.SiteType;
	const auto Field = SiteConfig.Field;
	const auto Helio = SiteConfig.HelioDesign;
	const auto AimStrategy = SiteConfig.AimStrategy;
	const auto Datetime = SiteConfig.DateTime;

	return DoAnalysis(CspSiteType, Field, Helio, AimStrategy, Datetime, OutResult);
}


void AIrradianceMode::DoAnalysisAsync(const ECspSite CspSiteType,
                                      const FSimpleField& Field,
                                      const FHeliostatDesign& HeliostatDesign,
                                      const FAimStrategy& AimStrategy,
                                      const FDateInfo& Datetime)
{
	// Clean up any existing task
	CleanupAnalysisTask();

	// Mark as running
	bAnalysisRunning = true;

	// Create and start the analysis task
	CurrentAnalysisTask = new FAnalysisTask(this, CspSiteType, Field, HeliostatDesign, AimStrategy, Datetime);
	AnalysisThread = FRunnableThread::Create(CurrentAnalysisTask, TEXT("FFIAM Analysis Thread"));
}


void AIrradianceMode::DoAnalysisFromPresetSiteAsync(const int SiteIndex)
{
	if (SiteIndex < 0 || SiteIndex >= LoadedSiteConfigs.Num())
	{
		UE_LOG(LogFlux, Error, TEXT("Invalid site index %d"), SiteIndex);
		OnAnalysisComplete.Broadcast(false);
		return;
	}

	const auto SiteConfig = LoadedSiteConfigs[SiteIndex];
	DoAnalysisAsync(SiteConfig.SiteType, SiteConfig.Field, SiteConfig.HelioDesign,
	                SiteConfig.AimStrategy, SiteConfig.DateTime);
}


void AIrradianceMode::CleanupAnalysisTask()
{
	if (AnalysisThread)
	{
		// Stop the thread if it's still running
		if (CurrentAnalysisTask)
		{
			CurrentAnalysisTask->Stop();
		}

		// Wait for thread to complete
		AnalysisThread->WaitForCompletion();
		delete AnalysisThread;
		AnalysisThread = nullptr;
	}

	if (CurrentAnalysisTask)
	{
		// Store the result if successful
		if (CurrentAnalysisTask->IsComplete() && CurrentAnalysisTask->WasSuccessful())
		{
			LastAnalysisResult = CurrentAnalysisTask->GetResult();
		}

		delete CurrentAnalysisTask;
		CurrentAnalysisTask = nullptr;
	}

	bAnalysisRunning = false;
}


