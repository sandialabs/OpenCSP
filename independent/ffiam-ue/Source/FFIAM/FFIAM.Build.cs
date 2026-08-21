using System.IO;
using UnrealBuildTool;

public class FFIAM : ModuleRules
{
	private string ProjectRootPath => Path.Combine(ModuleDirectory, "");

	public FFIAM(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		bEnableExceptions = true; // nlohmann/json uses exceptions

		PublicDependencyModuleNames.AddRange(new string[] {
			"Core",
			"CoreUObject",
			"Engine",
			"Json",
			"JsonUtilities",
			"InputCore",
			"SunPosition",
			// HUD-toggle hotkey: UMG for UUserWidget/GetAllWidgetsOfClass;
			// Slate/SlateCore/ApplicationCore for the global input pre-processor.
			"UMG",
			"Slate",
			"SlateCore",
			"ApplicationCore"
		});

		// FFIAM C++/CUDA headers. FFIAM_PATH = the FFIAM source root (include/,
		// src/, external/). Build FFIAM first: cmake -B build && cmake --build build.
		var ffiamPath = System.Environment.GetEnvironmentVariable("FFIAM_PATH");
		if (string.IsNullOrEmpty(ffiamPath))
		{
			// Windows default kept for back-compat; Linux/macOS should set FFIAM_PATH.
			ffiamPath = "T:/ffiam/ffiam/ffiam";
			System.Console.WriteLine($"FFIAM: FFIAM_PATH not set, using default: {ffiamPath}");
		}

		PublicIncludePaths.AddRange(new string[] {
			Path.Combine(ffiamPath, "include"),
			Path.Combine(ffiamPath, "src"),
			Path.Combine(ffiamPath, "external"),
			Path.Combine(ffiamPath, "external/fmt/include")
		});

		// Project root is two levels up from Source/FFIAM/.
		var projectDir = Path.GetFullPath(Path.Combine(ModuleDirectory, "../../"));

		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			ConfigureWindows(ffiamPath, projectDir);
		}
		else if (Target.Platform == UnrealTargetPlatform.Linux)
		{
			ConfigureLinux(ffiamPath, projectDir);
		}
		else
		{
			System.Console.WriteLine($"FFIAM: WARNING - platform {Target.Platform} not configured for the FFIAM/CUDA libraries.");
		}
	}

	// Windows: link ffiam_lib.lib, stage ffiam_lib.dll, link cudart.lib from CUDA_PATH.
	private void ConfigureWindows(string ffiamPath, string projectDir)
	{
		var cudaPath = System.Environment.GetEnvironmentVariable("CUDA_PATH");
		if (string.IsNullOrEmpty(cudaPath))
		{
			System.Console.WriteLine("FFIAM: CUDA_PATH not set; using default CUDA v12.8 path.");
			cudaPath = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8";
		}
		PublicIncludePaths.Add(Path.Combine(cudaPath, "include"));
		PublicAdditionalLibraries.Add(Path.Combine(cudaPath, "lib/x64", "cudart.lib"));

		var dllSource = Path.Combine(ffiamPath, "build_vs/bin/Release/ffiam_lib.dll");
		var libSource = Path.Combine(ffiamPath, "build_vs/lib/Release/ffiam_lib.lib");
		var binDir = Path.Combine(projectDir, "Binaries/Win64");
		var libDir = Path.Combine(ProjectRootPath, "Lib");
		Directory.CreateDirectory(binDir);
		Directory.CreateDirectory(libDir);

		if (File.Exists(dllSource))
			File.Copy(dllSource, Path.Combine(binDir, "ffiam_lib.dll"), true);
		else
			System.Console.WriteLine($"FFIAM: WARNING - ffiam_lib.dll not found at {dllSource}. Build FFIAM first.");

		if (File.Exists(libSource))
		{
			File.Copy(libSource, Path.Combine(libDir, "ffiam_lib.lib"), true);
			PublicAdditionalLibraries.Add(Path.Combine(libDir, "ffiam_lib.lib"));
		}
		else
		{
			System.Console.WriteLine($"FFIAM: WARNING - ffiam_lib.lib not found at {libSource}. Build FFIAM first.");
		}
	}

	// Linux: stage + link libffiam_lib.so (built via CMake). The module only uses
	// cudaError_t at compile time, so we don't link cudart here; the runtime lives
	// in the .so, keeping it decoupled from the host CUDA version.
	private void ConfigureLinux(string ffiamPath, string projectDir)
	{
		var cudaPath = System.Environment.GetEnvironmentVariable("CUDA_PATH");
		if (string.IsNullOrEmpty(cudaPath))
			cudaPath = "/usr/local/cuda"; // standard CUDA install location
		PublicIncludePaths.Add(Path.Combine(cudaPath, "include"));
		// Lets the linker resolve the .so's transitive libcudart dep.
		PublicSystemLibraryPaths.Add(Path.Combine(cudaPath, "lib64"));

		// Find libffiam_lib.so. Standard CMake output is $FFIAM_PATH/build/lib;
		// FFIAM_LIB overrides with a full path.
		string[] candidates = {
			System.Environment.GetEnvironmentVariable("FFIAM_LIB") ?? "",
			Path.Combine(ffiamPath, "build/lib/libffiam_lib.so"),
			Path.Combine(ffiamPath, "build_host/lib/libffiam_lib.so")
		};
		string soSource = null;
		foreach (var c in candidates)
		{
			if (!string.IsNullOrEmpty(c) && File.Exists(c)) { soSource = c; break; }
		}

		var binDir = Path.Combine(projectDir, "Binaries/Linux");
		Directory.CreateDirectory(binDir);

		if (soSource != null)
		{
			var soDest = Path.Combine(binDir, "libffiam_lib.so");
			File.Copy(soSource, soDest, true);
			PublicAdditionalLibraries.Add(soDest);
			RuntimeDependencies.Add(soDest);
		}
		else
		{
			System.Console.WriteLine("FFIAM: WARNING - libffiam_lib.so not found. Set FFIAM_PATH (and build FFIAM via CMake) or FFIAM_LIB.");
		}
	}
}
