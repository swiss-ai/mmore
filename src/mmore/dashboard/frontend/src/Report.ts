class Report {
    id: string
    workerId: string;
    finishedFilePaths: [string];
    timestamp: Date;

    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-expect-error
    constructor(data) {
        this.id = data._id;
        this.workerId = data.worker_id
        this.finishedFilePaths = data.finished_file_paths;
        this.timestamp = new Date(data.timestamp);
    }
}

export default Report;