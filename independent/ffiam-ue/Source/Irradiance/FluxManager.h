// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include "CoreMinimal.h"
#include "LogStream.h"
#include "SiteConfigTypes.h"
#include "GameFramework/Actor.h"
#include "SunPosition.h"
#include "FluxManager.generated.h"

class UHierarchicalInstancedStaticMeshComponent;
class UStaticMesh;

DECLARE_DYNAMIC_MULTICAST_DELEGATE_OneParam(FOnProcessingCompleteDelegate, bool, bSuccess);


UCLASS()
class IRRADIANCE_API AFluxManager : public AActor
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
	UHierarchicalInstancedStaticMeshComponent* VoxelHism ;
	
	UPROPERTY(EditAnywhere, Category = "Voxel")
	UStaticMesh* VoxelMesh;
	
	UPROPERTY(EditAnywhere, Category = "Voxel")
	UHierarchicalInstancedStaticMeshComponent* VoxelMedHism ;
	
	UPROPERTY(EditAnywhere, Category = "Voxel")
	UStaticMesh* VoxelMedMesh;
	
	UPROPERTY(EditAnywhere, Category = "Heliostats")
	UHierarchicalInstancedStaticMeshComponent* HelioHism ;
	
	UPROPERTY(EditAnywhere, Category = "Voxel")
	UStaticMesh* HelioMesh;
	
	UPROPERTY(EditAnywhere, Category = "Field")
	UStaticMeshComponent* AimPointComponent ;
	
	UPROPERTY(EditAnywhere, Category = "Field")
	float HelioSide = 90.0f;
	
	UPROPERTY(BlueprintAssignable, Category = "Analysis")
	FOnProcessingCompleteDelegate OnProcessingComplete;  // triggers result update
	
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	bool DoAnalysis();

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
	            float AimRingParam1, float AimRingParam2,
	            float AimVecParam1, float AimVecParam2, float AimVecParam3);
	
	UFUNCTION(BlueprintCallable, Category = "Analysis")
	void SetThresholds(float Thres1, float Thres2);

	LogStream Stream;

	bool AnalysisDone;
	
	int VoxelSize = 2;  // [m]

	// Max HISM instances before voxels get culled by priority. Prevents HISM from choking on large sites.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Voxel")
	int32 MaxVoxelInstances = 1000000;

	// Radius around aim point (m) where individual 2m voxels render. Beyond this, voxels merge into larger blocks.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Voxel")
	float DetailZoneRadius = 200.0f;

	// Merge factor for voxels outside the detail zone (e.g. 4 = merge 4x4x4 blocks into one instance at 4x scale)
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Voxel", meta = (ClampMin = "2", ClampMax = "8"))
	int32 MergeFactor = 4;

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


	// Converts 1D voxel index to world-space position (m). Grid is centered at origin in XY.
	static float3 GetVoxelLocFromIndex(const int VoxelIndex, const int VoxelSize, const int FieldRadius, const int MinHeight)
	{
		const int VoxelsPerSide = (FieldRadius * 2) / VoxelSize;
		const int VoxelsPerPlane = VoxelsPerSide * VoxelsPerSide;

		int3 GridCoords;
		GridCoords.z = VoxelIndex / VoxelsPerPlane;
		const int IndexInPlane = VoxelIndex - (GridCoords.z * VoxelsPerPlane);
		GridCoords.y = IndexInPlane / VoxelsPerSide;
		GridCoords.x = IndexInPlane - (GridCoords.y * VoxelsPerSide);

		const float fVs = static_cast<float>(VoxelSize);
		float3 WorldCoords;
		WorldCoords.x = GridCoords.x * fVs - static_cast<float>(FieldRadius);
		WorldCoords.y = GridCoords.y * fVs - static_cast<float>(FieldRadius);
		WorldCoords.z = GridCoords.z * fVs + static_cast<float>(MinHeight);

		return WorldCoords;
	}

};
