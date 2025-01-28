import {useEffect, useState} from "react";
import "./App.css";
import {WorkerLatest} from "./WorkerLatest.ts";
import "chartjs-adapter-date-fns";
import ReportsTableView from "./ReportsTableView.tsx";
import GraphAndSublistView from "./GraphAndSublistView.tsx";
import ReportsChartView from "./ReportsChartView.tsx";
import OverallProgressView from "./OverallProgressView.tsx";

const apiUrl = import.meta.env.VITE_BACKEND_API_URL;

function App() {
    const [workers, setWorkers] = useState<WorkerLatest[]>([]);

    function loadWorkersLatest() {
        fetch(apiUrl + "/reports/workers/latest")
            .then((res) => res.json())
            .then((data) => {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                const workers = data.map((worker) => new WorkerLatest(worker));
                setWorkers(workers);
            })
            .catch((error) => {
                console.error("Error fetching workers latest:", error);
            });
    }

    useEffect(() => {
        loadWorkersLatest();
    }, []);


    return (
        <div className="dashboard">
            <header className="header">
                <h1>MMORE Dashboard 🐮🚀️</h1>
            </header>

            <section className="system-overview">
                <OverallProgressView/>
                <ReportsChartView workers={workers}/>
            </section>

            <GraphAndSublistView workers={workers}/>

            <ReportsTableView/>
        </div>
    );
}

export default App;
