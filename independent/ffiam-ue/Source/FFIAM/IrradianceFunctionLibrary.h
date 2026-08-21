// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "ffiam/ffiam.h"
#include "IrradianceFunctionLibrary.generated.h"


class AIrradianceMode;
struct FSiteConfig;

UCLASS()
class FFIAM_API UIrradianceFunctionLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category = "Game", meta = (WorldContext = "WorldContextObject"))
	static AIrradianceMode* GetGameMode(const UObject* WorldContextObject);
	
	
	/**
	 * Loads a single FFIAM site configuration, with optional aim params, from the specified JSON file.
	 * The JSON file should contain a single JSON object that maps to the FSiteConfig USTRUCT.
	 *
	 * @param FilePath The path to the JSON configuration file. Can be absolute or relative to Project Content directory.
	 * @param OutSiteConfig The FSiteConfig USTRUCT instance to populate with data from the JSON file.
	 * @return True if the configuration was successfully loaded and parsed, false otherwise.
	 */
	UFUNCTION(BlueprintCallable, Category = "FFIAM|Configuration")
	static bool LoadSiteConfigFromJson(const FString& FilePath, UPARAM(ref) FSiteConfig& OutSiteConfig);
	
	/**
	 * Loads all FFIAM site configurations from JSON files found in a specified directory.
	 * Note that JSON files will not appear in UE Content Browser!
	 *
	 * @param RelativeDirectoryPath The directory path relative to the Project Content directory (e.g., "Data/SiteConfigs").
	 * @param OutLoadedConfigs An array to be populated with the successfully loaded FSiteConfigUE structs.
	 * @return The number of configurations successfully loaded.
	 */
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	static int32 LoadAllSiteConfigsFromDirectory(const FString& RelativeDirectoryPath, TArray<FSiteConfig>& OutLoadedConfigs);


private:
	// Helper to map string names to enum values from ffiam.h
	static aim_strategy_type StringToAimStrategyType(const FString& AimStratStr);
	static csp_site StringToCspSite(const FString& SiteStr);
};
