import {Line} from "react-chartjs-2";
import {
    CategoryScale,
    Chart as ChartJS,
    Legend,
    LinearScale,
    LineElement,
    PointElement,
    TimeScale,
    Title,
    Tooltip,
} from "chart.js";
import "./App.css";
import {WorkerLatest} from "./WorkerLatest.ts";
import "chartjs-adapter-date-fns";

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
    TimeScale,
);

function ReportsChartView({workers}: { workers: WorkerLatest[] }) {

    function cumulative(workers: WorkerLatest[]): { x: Date; y: number; count: number; isTotalDataset: boolean }[] {
        const cumulativeData: {
            x: Date;
            y: number;
            count: number;
            isTotalDataset: boolean;
        }[] = [];

        const allReports = workers
            .flatMap((worker) =>
                worker.latestReports.map((report) => ({
                    timestamp: report.timestamp,
                    count: report.count,
                })),
            )
            .sort((a, b) => +a.timestamp - +b.timestamp);

        let cumulative = 0;
        allReports.forEach((report) => {
            cumulative += report.count;
            cumulativeData.push({
                x: report.timestamp,
                y: cumulative,
                count: report.count,
                isTotalDataset: true,
            });
        });
        return cumulativeData;
    }

    function workerColor(workerId: string): string {
        const str: string = "yeah" + workerId + "increase"; // more variation in the colours

        // piece of code copied from https://stackoverflow.com/a/16348977/7974174
        let hash = 0;
        str.split('').forEach(char => {
            hash = char.charCodeAt(0) + ((hash << 5) - hash)
        })
        let colour = '#'
        for (let i = 0; i < 3; i++) {
            const value = (hash >> (i * 8)) & 0xff
            colour += value.toString(16).padStart(2, '0')
        }

        return colour
    }

    function cumulativePerWorker(workers: WorkerLatest[]): {
        label: string;
        data: { x: Date; y: number; count: number }[];
        borderColor: string;
        fill: boolean
    }[] {
        const perWorkerCumulative = workers.map((worker) => ({
            label: `Worker ${worker.workerId}`,
            data: worker.latestReports.map((report) => ({
                x: report.timestamp,
                y: report.count,
                count: report.count,
            })),
            borderColor: workerColor(worker.workerId),
            fill: false,
        }));

        for (const worker of perWorkerCumulative) {
            let cumulative = 0;
            worker.data.sort().reverse();
            for (const item of worker.data) {
                cumulative += item.y;
                item.y = cumulative;
            }
        }
        return perWorkerCumulative;
    }

    function datasets() {
        const joinedWorkers = {
            label: "All Workers",
            data: cumulative(workers),
            borderColor: "green",
            fill: false,
        };

        const perWorkerCumulative = cumulativePerWorker(workers);

        return [joinedWorkers, ...perWorkerCumulative];
    }

    return (
        <div className="task-chart">
            <h3>Time Evolution of Jobs</h3>
            <Line
                data={{
                    datasets: datasets(),
                }}
                options={{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: {
                            position: "top",
                        },
                        tooltip: {
                            callbacks: {
                                label: function (tooltipItem) {
                                    const raw = tooltipItem.raw;
                                    const total = tooltipItem.formattedValue;
                                    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                                    // @ts-ignore
                                    if (raw.isTotalDataset) {
                                        return `Total: ${total}`;
                                    }
                                    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                                    // @ts-ignore
                                    return `New items: ${raw.count} Worker total: ${total} `;

                                },
                            },
                        },
                    },
                    scales: {
                        x: {
                            type: "time",
                            time: {
                                unit: "minute",
                                tooltipFormat: "HH:mm:ss",
                                displayFormats: {
                                    minute: "HH:mm:ss",
                                },
                            },
                            title: {
                                display: true,
                                text: "Time",
                            },
                        },
                        y: {
                            title: {
                                display: true,
                                text: "Count",
                            },
                        },
                    },
                }}
                height={200}
            />
            <p>[notice] For performance reasons of the frontend this chart displays only based on the 1000 last reports
                made by each worker</p>
        </div>
    );
}

export default ReportsChartView;
