#pragma once

#include <driver_types.h>
#include <vector_types.h>
#include "ffiam/ffiam.h"

#include "CoreMinimal.h"
#include "SiteConfigTypes.generated.h" // Must be last include


// --- UENUM Definitions ---

UENUM(BlueprintType)
// Values must match the C++ csp_site enum (cast directly in IrradianceMode).
enum class ECspSite : uint8
{
    Site_Custom UMETA(DisplayName="Custom Site"),
    Site_Nsttf UMETA(DisplayName="NSTTF"),
    Site_RadialSmall UMETA(DisplayName="Radial (1 km)"),
    Site_Radial UMETA(DisplayName="Radial (1.6 km)"),
    Site_SampleV1 UMETA(DisplayName="Sample V1"),
    Site_SampleV2 UMETA(DisplayName="Sample V2"),
    Site_SampleV3 UMETA(DisplayName="Sample V3")
};


UENUM(BlueprintType)
enum class EAimStrategyType : uint8
{
    Aim_Null UMETA(DisplayName="Null"),
    Aim_Point UMETA(DisplayName="Point"),
    Aim_Ring UMETA(DisplayName="Ring"),
    Aim_Split_Ring UMETA(DisplayName="Split Ring"),
    Aim_Vector UMETA(DisplayName="Vector"),
    Aim_Data UMETA(DisplayName="Data")
};


// --- USTRUCT Definitions ---

USTRUCT(BlueprintType)
struct FFieldVector // Helper for params in FAimStrategy if it's a vector
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM")
    float X;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM")
    float Y;
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM")
    float Z;

    FFieldVector() : X(0.f), Y(0.f), Z(0.f) {}

    // convert to ffiam.h::float3
    operator float3() const { return {X, Y, Z}; }
    
    // Constructor from ffiam.h::float3
    FFieldVector(const float3& InVec) : X(InVec.x), Y(InVec.y), Z(InVec.z) {}
};


USTRUCT(BlueprintType)
struct FHeliostatDesign
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Heliostat")
    int32 NumFacets;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Heliostat")
    int32 NumRows;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Heliostat")
    int32 NumCols;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Heliostat")
    float FacetWidth;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Heliostat")
    float FacetHeight;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Heliostat")
    FString FacetFile; // std::string maps to FString

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Heliostat")
    bool bFacetsImported;

    FHeliostatDesign()
        : NumFacets(35), NumRows(7), NumCols(5),
          FacetWidth(1.8f), FacetHeight(1.8f),
          bFacetsImported(false)
    {}
};

USTRUCT(BlueprintType)
struct FDateInfo
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Time")
    int32 Year;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Time")
    int32 Month;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Time")
    int32 Day;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Time")
    int32 Hour;

    FDateInfo() : Year(2025), Month(6), Day(21), Hour(13) {}
};


USTRUCT(BlueprintType)
struct FAimStrategy
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Aiming")
    EAimStrategyType AimStrategyType; // Enum

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Aiming")
    FFieldVector Params;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Aiming")
    FString ImportFile;

    FAimStrategy() : AimStrategyType(EAimStrategyType::Aim_Point) {}
};


/** CSP field parameters (no allocated arrays) */
USTRUCT(BlueprintType)
struct FSimpleField
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    float Latitude;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    float Longitude;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    float Timezone;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    int32 Radius;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    int32 MinHeight;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    int32 MaxHeight;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    FFieldVector TowerPos;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    float TowerHeight;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    int32 VoxelSize;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    int32 NumVoxels;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    float VoxelArea;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    int32 NumHelios;
    
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    FString HelioCoordinateFile;

    // Field generation when HelioCoordinateFile is empty: "grid" or "radial".
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Field")
    FString Layout;

    FSimpleField()
        : Latitude(0.f), Longitude(0.f), Timezone(0.f),
          Radius(1000), MinHeight(4), MaxHeight(200),
          TowerHeight(100.f), VoxelSize(2), NumVoxels(0), VoxelArea(0.f),
          NumHelios(0), Layout(TEXT("grid"))
    {}
};


/** Top-level site config for analysis. Memory management stays in Mode. */
USTRUCT(BlueprintType)
struct FSiteConfig
{
    GENERATED_BODY()

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Config")
    FString RunId;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Config")
    FHeliostatDesign HelioDesign;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Config")
    FDateInfo DateTime;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Config")
    ECspSite SiteType;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Config")
    FAimStrategy AimStrategy;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "FFIAM Config")
    FSimpleField Field;

    FSiteConfig() : SiteType(ECspSite::Site_Custom) {}
};


/**
 * Analysis results, including pointers to memory blocks. Memory must be handled inside Mode!
 * Treat this struct as read-only in other objects.
 */
USTRUCT(BlueprintType)
struct FAnalysisResult
{
    GENERATED_BODY()

	cudaError_t CudaStatus;
	float* Irrads = nullptr;
	float* MiscData = nullptr;
    float3* Misc3Data = nullptr;
    
    heliostat* Helios = nullptr;
    // heliostat data-as-arrays; used to return data to python
    float3* Locs = nullptr;
    float3* AimVs = nullptr;
    float* MoveAngles = nullptr;
};


