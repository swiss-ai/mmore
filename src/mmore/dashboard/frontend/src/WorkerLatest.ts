class LatestReport {
    timestamp: Date;
    count: number;

    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-expect-error
    constructor(data) {
        this.timestamp = new Date(data.timestamp);
        this.count = data.count;
    }
}

class WorkerLatest {
    workerId: string;
    latestTimestamp: Date;
    lastActive: string;
    latestReports: LatestReport[];

    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-expect-error
    constructor(data) {
        this.workerId = data.worker_id;
        this.latestTimestamp = new Date(data.latest_timestamp);
        this.lastActive = data.last_active;

        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-expect-error
        this.latestReports = data.latest_reports.map((report) => new LatestReport(report));
    }

    status(): string {
        const now = new Date();
        const diff = now.getTime() - this.latestTimestamp.getTime();
        if (diff > 60 * 60 * 1000) {
            return 'error';
        } else if (diff > 30 * 60 * 1000) {
            return 'warning';
        } else {
            return 'active';
        }
    }
}

export {WorkerLatest, LatestReport};