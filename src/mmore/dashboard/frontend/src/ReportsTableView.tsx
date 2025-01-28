import {useEffect, useState} from "react";
import {Column} from "primereact/column";
import {DataTable} from "primereact/datatable";
import "./App.css";
import "chartjs-adapter-date-fns";
import Report from "./Report.ts";

const apiUrl = import.meta.env.VITE_BACKEND_API_URL;

function ReportsTableView() {
    const [reports, setReports] = useState<Report[]>([]);
    const [page, setPage] = useState(0);
    const [pageSize, setPageSize] = useState(5);
    const [totalRecords, setTotalRecords] = useState(0);
    const [loading, setLoading] = useState(true);

    function loadReports() {
        setLoading(true);
        fetch(apiUrl + "/reports/latest/?page_idx=" + page + "&page_size=" + pageSize)
            .then((res) => res.json())
            .then((data) => {
                // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                // @ts-expect-error
                const reports = data.reports.map((report) => new Report(report));

                setReports(reports);
                setTotalRecords(data.total_records);
                setLoading(false);
            })
            .catch((error) => {
                console.error("Error fetching reports:", error);
            });
    }

    useEffect(() => {
        loadReports();
    }, [page, pageSize]);

    return (
        <section className="logs">
            <h3>Activity Logs</h3>
            <p>Total record {totalRecords}</p>
            <DataTable
                value={reports}
                paginator={true}
                rows={pageSize}
                first={page * pageSize}
                totalRecords={totalRecords}
                lazy={true}
                onPage={(e) => {
                    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
                    // @ts-expect-error
                    setPage(e.page);
                    setPageSize(e.rows);
                }}
                loading={loading}
                rowsPerPageOptions={[5, 10, 20, 50, 100]}
                showGridlines={true}
            >
                <Column field="timestamp" header="Timestamp" body={(rowData) => rowData.timestamp.toLocaleString()}/>
                <Column field="workerId" header="Worker id"/>
                <Column field="finishedFilePaths" header="Files" body={(rowData) => {
                    return (
                        <ul>
                            {rowData.finishedFilePaths.map((file: string, index: number) => (
                                <li key={index}>{file}</li>
                            ))}
                        </ul>
                    );
                }}/>
            </DataTable>
        </section>
    )
}

export default ReportsTableView;
