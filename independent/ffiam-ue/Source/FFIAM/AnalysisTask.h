// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

#pragma once

#include "CoreMinimal.h"
#include "SiteConfigTypes.h"


class AIrradianceMode;
/** Background thread for FFIAM analysis */
class FAnalysisTask : public FRunnable
{
public:
	FAnalysisTask(AIrradianceMode* InMode,
	              const ECspSite InCspSiteType,
	              const FSimpleField& InField,
	              const FHeliostatDesign& InHeliostatDesign,
	              const FAimStrategy& InAimStrategy,
	              const FDateInfo& InDatetime);

	virtual ~FAnalysisTask() override;

	// FRunnable interface
	virtual bool Init() override;
	virtual uint32 Run() override;
	virtual void Stop() override;
	virtual void Exit() override;

	bool IsComplete() const { return bIsComplete; }
	bool WasSuccessful() const { return bSuccess; }
	FAnalysisResult GetResult() const { return Result; }

private:
	AIrradianceMode* Mode;
	ECspSite CspSiteType;
	FSimpleField Field;
	FHeliostatDesign HeliostatDesign;
	FAimStrategy AimStrategy;
	FDateInfo Datetime;
	FAnalysisResult Result;

	FThreadSafeBool bIsComplete;
	FThreadSafeBool bSuccess;
	FThreadSafeBool bShouldStop;
};
