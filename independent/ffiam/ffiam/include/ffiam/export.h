#pragma once

#ifdef _WIN32
    #ifdef FFIAM_EXPORTS
        #define FFIAM_API __declspec(dllexport)
    #else
        #define FFIAM_API __declspec(dllimport)
    #endif
#else
    #define FFIAM_API
#endif
