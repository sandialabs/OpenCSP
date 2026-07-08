// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#include "IrradianceFunctionLibrary.h"

#include "IrradianceMode.h"
#include "LogChannels.h"
#include "SiteConfigTypes.h"
#include "GameFramework/GameModeBase.h"
#include "HAL/PlatformFileManager.h"
#include "JsonObjectConverter.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"



aim_strategy_type UIrradianceFunctionLibrary::StringToAimStrategyType(const FString& AimStratStr)
{
    if (AimStratStr.Equals(TEXT("Point"), ESearchCase::IgnoreCase)) return aim_point;
    if (AimStratStr.Equals(TEXT("Ring"), ESearchCase::IgnoreCase)) return aim_ring;
    if (AimStratStr.Equals(TEXT("SplitRing"), ESearchCase::IgnoreCase)) return aim_split_ring;
    if (AimStratStr.Equals(TEXT("Vector"), ESearchCase::IgnoreCase)) return aim_vector;
    if (AimStratStr.Equals(TEXT("Data"), ESearchCase::IgnoreCase)) return aim_data_csv;
    UE_LOG(LogFlux, Warning, TEXT("Unknown Aim Strategy Type string: %s. Defaulting to aim_null."), *AimStratStr);
    return aim_null;
}

csp_site UIrradianceFunctionLibrary::StringToCspSite(const FString& SiteStr)
{
    if (SiteStr.Equals(TEXT("NSTTF"), ESearchCase::IgnoreCase)) return site_nsttf;
    if (SiteStr.Equals(TEXT("RadialSmall"), ESearchCase::IgnoreCase)) return site_radialSmall;
    if (SiteStr.Equals(TEXT("Radial"), ESearchCase::IgnoreCase)) return site_radial;
    if (SiteStr.Equals(TEXT("SampleV1"), ESearchCase::IgnoreCase)) return site_sampleV1;
    if (SiteStr.Equals(TEXT("SampleV2"), ESearchCase::IgnoreCase)) return site_sampleV2;
    if (SiteStr.Equals(TEXT("SampleV3"), ESearchCase::IgnoreCase)) return site_sampleV3;
    UE_LOG(LogFlux, Warning, TEXT("Unknown CSP Site string: %s. Defaulting to site_null."), *SiteStr);
    return site_null;
}


AIrradianceMode* UIrradianceFunctionLibrary::GetGameMode(const UObject* WorldContextObject)
{
    if (!WorldContextObject) return nullptr;

    UWorld* World = GEngine->GetWorldFromContextObjectChecked(WorldContextObject);
    return World ? Cast<AIrradianceMode>(World->GetAuthGameMode()) : nullptr;
}



bool UIrradianceFunctionLibrary::LoadSiteConfigFromJson(const FString& FilePath, FSiteConfig& OutSiteConfig)
{
	FString FullPath = FilePath;
	if (FPaths::IsRelative(FullPath))
	{
		FullPath = FPaths::ProjectContentDir() / FullPath;
	}
	FullPath = FPaths::ConvertRelativePathToFull(FullPath);

	if (!FPlatformFileManager::Get().GetPlatformFile().FileExists(*FullPath))
	{
		UE_LOG(LogFlux, Error, TEXT("Site Config UE JSON file not found: %s"), *FullPath);
		return false;
	}

	FString JsonString;
	if (!FFileHelper::LoadFileToString(JsonString, *FullPath))
	{
		UE_LOG(LogFlux, Error, TEXT("Failed to load Site Config UE JSON to string: %s"), *FullPath);
		return false;
	}

	TSharedPtr<FJsonObject> RootJsonObject;
	TSharedRef<TJsonReader<>> JsonReader = TJsonReaderFactory<>::Create(JsonString);

	if (!FJsonSerializer::Deserialize(JsonReader, RootJsonObject) || !RootJsonObject.IsValid())
	{
		UE_LOG(LogFlux, Error, TEXT("Failed to deserialize Site Config UE JSON: %s. Error: %s"), *FullPath, *JsonReader->GetErrorMessage());
		return false;
	}

	if (FJsonObjectConverter::JsonObjectToUStruct(RootJsonObject.ToSharedRef(), FSiteConfig::StaticStruct(), &OutSiteConfig, 0, 0))
	{
		UE_LOG(LogFlux, Log, TEXT("Successfully loaded FSiteConfig from JSON: %s. RunId: %s"), *FullPath, *OutSiteConfig.RunId);
		// Calculate derived fields not in JSON.
	    auto& Field = OutSiteConfig.Field;
	    auto& HelioDesign = OutSiteConfig.HelioDesign;
        HelioDesign.NumRows = HelioDesign.NumFacets / HelioDesign.NumCols;
	    
        Field.VoxelArea = static_cast<int>(pow(Field.VoxelSize, 2));
        const int NumVoxelsZ = (Field.MaxHeight - Field.MinHeight) / Field.VoxelSize;
        const int NumVoxelsY = 2 * Field.Radius / Field.VoxelSize;
        const int NumVoxelsX = 2 * Field.Radius / Field.VoxelSize;
        Field.NumVoxels = NumVoxelsX * NumVoxelsY * NumVoxelsZ;
	    
		return true;
	}
	
	UE_LOG(LogFlux, Error, TEXT("Failed to convert RootJsonObject to FSiteConfigUE for: %s"), *FullPath);
	return false;
}




int32 UIrradianceFunctionLibrary::LoadAllSiteConfigsFromDirectory(const FString& RelativeDirectoryPath,
                                                                  TArray<FSiteConfig>& OutLoadedConfigs)
{
	OutLoadedConfigs.Empty();

    FString FullDirectoryPath = FPaths::ProjectConfigDir() / RelativeDirectoryPath;
    FullDirectoryPath = FPaths::ConvertRelativePathToFull(FullDirectoryPath);
    FPaths::NormalizeDirectoryName(FullDirectoryPath); // Ensure it ends with a slash for FindFiles

    UE_LOG(LogFlux, Log, TEXT("Searching for JSON config files in directory: %s"), *FullDirectoryPath);

    TArray<FString> FoundFiles;
    FPlatformFileManager::Get().GetPlatformFile().FindFiles(FoundFiles, *FullDirectoryPath, TEXT(".json"));

    if (FoundFiles.IsEmpty())
    {
        UE_LOG(LogFlux, Warning, TEXT("No .json files found in directory: %s"), *FullDirectoryPath);
        return 0;
    }

    int32 SuccessfulLoads = 0;
    for (const FString& FileName : FoundFiles)
    {
        UE_LOG(LogFlux, Log, TEXT("Attempting to load config from: %s"), *FileName);

        FSiteConfig LoadedConfig;
        if (LoadSiteConfigFromJson(FileName, LoadedConfig)) // LoadSiteConfigUEFromJson expects a full or resolvable path
        {
            OutLoadedConfigs.Add(LoadedConfig);
            SuccessfulLoads++;
        }
    }

    UE_LOG(LogFlux, Log, TEXT("Finished loading configs. Successfully loaded %d out of %d files."), SuccessfulLoads, FoundFiles.Num());
    return SuccessfulLoads;
}
