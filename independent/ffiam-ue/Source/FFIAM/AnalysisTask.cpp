#include "AnalysisTask.h"

#include "IrradianceMode.h"


FAnalysisTask::FAnalysisTask(AIrradianceMode* InMode,
                             const ECspSite InCspSiteType,
                             const FSimpleField& InField,
                             const FHeliostatDesign& InHeliostatDesign,
                             const FAimStrategy& InAimStrategy,
                             const FDateInfo& InDatetime)
	: Mode(InMode)
	  , CspSiteType(InCspSiteType)
	  , Field(InField)
	  , HeliostatDesign(InHeliostatDesign)
	  , AimStrategy(InAimStrategy)
	  , Datetime(InDatetime)
	  , Result(), bIsComplete(false)
	  , bSuccess(false)
	  , bShouldStop(false)
{
}


FAnalysisTask::~FAnalysisTask()
{
}


bool FAnalysisTask::Init()
{
	return true;
}


uint32 FAnalysisTask::Run()
{
	// Report progress at 0%
	if (Mode && Mode->GetWorld())
	{
		FGraphEventRef Task = FFunctionGraphTask::CreateAndDispatchWhenReady(
			[this]()
			{
				Mode->OnAnalysisProgress.Broadcast(0.0f);
			},
			TStatId(), nullptr, ENamedThreads::GameThread);
	}

	// Run the analysis
	bSuccess = Mode->DoAnalysisInternal(CspSiteType, Field, HeliostatDesign, AimStrategy, Datetime, Result);

	// Report progress at 100% and completion
	if (Mode && Mode->GetWorld())
	{
		FGraphEventRef Task = FFunctionGraphTask::CreateAndDispatchWhenReady(
			[this]()
			{
				Mode->OnAnalysisProgress.Broadcast(100.0f);

				if (bSuccess)
				{
					Mode->LastAnalysisResult = Result;
				}
				Mode->bAnalysisRunning = false;
				Mode->OnAnalysisComplete.Broadcast(bSuccess);
			},
			TStatId(), nullptr, ENamedThreads::GameThread);
	}

	bIsComplete = true;
	return 0;
}


void FAnalysisTask::Stop()
{
	bShouldStop = true;
}


void FAnalysisTask::Exit()
{
}
