// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/GameState.h"
#include "IrradianceGameState.generated.h"

struct FSiteConfig;


/** Shares Mode data with UI */
UCLASS()
class IRRADIANCE_API AIrradianceGameState : public AGameState
{
	GENERATED_BODY()

public:
	UPROPERTY(Replicated, BlueprintReadOnly, Category = "Game State")
	TArray<FSiteConfig> LoadedSiteConfigs;

protected:
	virtual void GetLifetimeReplicatedProps(TArray<FLifetimeProperty>& OutLifetimeProps) const override;
};
