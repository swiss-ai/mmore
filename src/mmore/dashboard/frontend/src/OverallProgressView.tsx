import {useEffect, useState} from "react";
import "./App.css";
import Progress from "./Progress";
import "chartjs-adapter-date-fns";

const apiUrl = import.meta.env.VITE_BACKEND_API_URL;

function OverallProgressView() {
    const [progress, setProgress] = useState<Progress>();

    function loadProgress() {
        fetch(apiUrl + "/progress")
            .then((res) => res.json())
            .then((data) => {
                const progress = new Progress(data);
                setProgress(progress);
            })
            .catch((error) => {
                console.error("Error fetching progress:", error);
            });
    }

    function stopExecution() {
        fetch(apiUrl + "/stop", {
            method: "POST",
        })
            .then((res) => res.json())
            .then((data) => {
                console.log("Stop Execution:", data);
                loadProgress();
            })
            .catch((error) => {
                alert("Error stopping execution:" + error);
            });
    }

    useEffect(() => {
        loadProgress();
    }, []);

    return (
        <div className="progress">
            <h3>Overall Progress</h3>
            <div className="progress-circle">
                <div className="circle-inner">
                    <p className="progress-number">
                        {progress != null ? progress.progress.toFixed(2) : "loading"}%
                    </p>
                    <p className="progress-label">
                        {progress != null
                            ? progress.askToStop
                                ? "Stopping..."
                                : progress.progress === 100
                                    ? "Completed"
                                    : "In Progress"
                            : "loading"}
                    </p>
                </div>
            </div>
            <p>
                Total Files Processed:{" "}
                <strong>{progress != null ? progress.finishedFiles : "..."}</strong>
            </p>
            <p>
                Total files to process:{" "}
                <strong>{progress != null ? progress.totalFiles : "..."}</strong>
            </p>
            <p>
                Start time :{" "}
                <strong>
                    {progress != null ? progress.startTime.toLocaleString() : "..."}
                </strong>
            </p>
            <p>
                Latest update :{" "}
                <strong>{progress != null ? progress.lastActivity : "..."}</strong>
            </p>
            <div className="controls">
                {progress != null && progress.askToStop ? (
                    <p>stopping...</p>
                ) : (
                    <button
                        className="stop"
                        onClick={() => {
                            const agree: boolean = confirm(
                                "Are you sure you want to stop the execution?",
                            );
                            if (agree) {
                                stopExecution();
                            }
                        }}
                    >
                        Stop Execution
                    </button>
                )}
            </div>
        </div>
    );
}

export default OverallProgressView;
