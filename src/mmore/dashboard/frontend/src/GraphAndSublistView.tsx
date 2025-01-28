import {useRef, useState} from "react";
import ForceGraph2D from "react-force-graph-2d";
import {Column} from "primereact/column";
import {DataTable} from "primereact/datatable";
import "./App.css";
import {WorkerLatest} from "./WorkerLatest.ts";

function GraphAndSublistView({workers}: { workers: WorkerLatest[] }) {
    // This component displays a graph of workers and when clicking on a node of this graph it scrolls to the corresponding worker card underneath the graph view.
    // workers is an array that was already fetched

    const [highlightedWorker, setHighlightedWorker] = useState<string | number>();

    const workerRefs = useRef({}); // refs for worker cards, for the scroll into view functionality

    const scrollToWorker = (workerId: string | number) => {
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-expect-error
        const ref = workerRefs.current[workerId];
        if (ref) {
            ref.scrollIntoView({behavior: "smooth", block: "center"});
            setHighlightedWorker(workerId);
        }
    };

    const graphData = {
        nodes: workers.map((worker) => ({
            id: worker.workerId,
            name: worker.workerId,
            status: worker.status(),
        })),
        links: workers.map((worker) => ({
            source: worker.workerId,
            target: workers.find((e) => e.workerId === "0") === undefined ? workers[0].workerId : "0"
        })),
    };

    const workerGraph = (
        <section className="worker-graph">
            <h3>Worker Graph</h3>
            {
                (workers.length === 0) ?
                    <p>Loading...</p> :
                    <ForceGraph2D
                        graphData={graphData}
                        nodeCanvasObject={(node, ctx) => {
                            ctx.fillStyle =
                                node.status === "active"
                                    ? "green"
                                    : node.status === "error"
                                        ? "red"
                                        : "gray";
                            ctx.beginPath();
                            ctx.arc(node.x || 0, node.y || 0, 8, 0, 2 * Math.PI);
                            ctx.fill();
                            ctx.font = "8px Arial";
                            ctx.fillStyle = "black";
                            ctx.fillText(node.name, (node.x || 0) + 10, (node.y || 0) + 3);
                            // put gear emoji in middle of node:
                            if (node.status === "active") {
                                ctx.fillText("⚙️", (node.x || 0) - 5, (node.y || 0) + 3);
                            } else if (node.status === "error") {
                                ctx.fillText("️⚠️", (node.x || 0) - 5, (node.y || 0) + 3);
                            }
                        }}
                        linkColor={() => "lightblue"}
                        width={650}
                        height={300}
                        onNodeClick={(node) => {
                            scrollToWorker(node.id);
                        }}
                    />
            }
        </section>
    );

    const workerStatus = (
        <section className="workers">
            <h3>Worker Status</h3>
            <div
                className="worker-list"
                style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                    gap: "20px",
                }}
            >
                {workers.map((worker) => (
                    <div
                        key={worker.workerId}
                        ref={(el) => { // attach ref to each worker card
                            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                            // @ts-expect-error
                            return (workerRefs.current[worker.workerId] = el);
                        }}
                        className={`worker-card ${worker.status()} ${highlightedWorker === worker.workerId ? "highlighted" : ""}`}
                        style={{
                            textAlign: "center",
                            border:
                                highlightedWorker === worker.workerId
                                    ? "2px solid blue"
                                    : "none",
                        }}
                    >
                        <h4>{worker.workerId}</h4>
                        <p title={worker.latestTimestamp.toLocaleString()}>
                            Last Active: {worker.lastActive}
                        </p>
                        <p>Latest batches of files processed:</p>
                        <DataTable
                            value={worker.latestReports}
                            showGridlines
                            stripedRows
                            paginator
                            size={"small"}
                            paginatorTemplate="PageLinks"
                            rows={4}
                        >
                            <Column
                                field="timestamp"
                                header="Timestamp"
                                body={(rowData) => rowData.timestamp.toLocaleString()}
                            />
                            <Column field="count" header="Count"/>
                        </DataTable>
                    </div>
                ))}
            </div>
        </section>
    );

    return (
        <>
            {workerGraph}

            {workerStatus}
        </>
    );
}

export default GraphAndSublistView;
