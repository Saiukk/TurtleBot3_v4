using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildPlayer
{
    public static void Build()
    {
        string[] scenes = { "Assets/Scenes/Training_arena.unity" };

        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = scenes,
            target = BuildTarget.StandaloneOSX,
            locationPathName = "../DQN/env/macos_training/SafeRobotics.app",
            options = BuildOptions.None
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        BuildSummary summary = report.summary;

        if (summary.result == BuildResult.Succeeded)
        {
            Debug.Log($"Build succeeded: {summary.totalSize} bytes");
        }
        else
        {
            Debug.LogError($"Build failed: {summary.result}");
            foreach (var step in report.steps)
            {
                foreach (var message in step.messages)
                {
                    if (message.type == LogType.Error || message.type == LogType.Exception)
                    {
                        Debug.LogError(message.content);
                    }
                }
            }
        }

        EditorApplication.Exit(summary.result == BuildResult.Succeeded ? 0 : 1);
    }
}
