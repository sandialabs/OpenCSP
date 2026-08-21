// Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.

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
