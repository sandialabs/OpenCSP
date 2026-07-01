// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#include "IrradianceMode.h"

#include "AnalysisTask.h"
#include "ffiam/ffiam.h"
#include "Irradiance.h"
#include "IrradianceFunctionLibrary.h"
#include "IrradianceGameState.h"
#include "LogChannels.h"
#include "HAL/PlatformFileManager.h" // For FPlatformFileManager
#include "Misc/Paths.h"             // For FPaths
#include "SiteConfigTypes.h"
#include "Async/TaskGraphInterfaces.h" // For FFunctionGraphTask


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


void AIrradianceMode::PrepMemoryForAnalysis()
{
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

	PrepMemoryForAnalysis();
	
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


