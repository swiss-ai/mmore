class Progress {
    totalFiles: number;
    startTime: Date;
    finishedFiles: number;
    progress: number;
    lastActivity: string;
    askToStop: boolean;

    // eslint-disable-next-line @typescript-eslint/ban-ts-comment
    // @ts-expect-error
    constructor(data) {
        this.totalFiles = data.total_files;
        this.startTime = new Date(data.start_time);
        this.finishedFiles = data.finished_files;
        this.progress = data.progress;
        this.lastActivity = data.last_activity;
        this.askToStop = data.ask_to_stop;
    }
}

export default Progress;