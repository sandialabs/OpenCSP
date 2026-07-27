#pragma once
#include <sstream>

// Redirects DLL stdout to UE log
class LogStream : public std::stringbuf
{
protected:
	int sync()
	{
		UE_LOG(LogTemp, Log, TEXT("%s"), *FString(str().c_str()));
		str("");
		return std::stringbuf::sync();
	}
};
