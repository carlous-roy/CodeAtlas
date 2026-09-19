package demo;

import java.util.List;

/** Creates report jobs and hands finished reports to the user. */
public class ReportService {

    private final StorageClient storage;

    public ReportService(StorageClient storage) {
        this.storage = storage;
    }

    /** Where a finished report can be downloaded from. */
    public String getDownloadUrl(ReportJob job) {
        return storage.presign(job.getKey());
    }

    public ReportJob submit(ReportType type) {
        return new ReportJob(type);
    }

    public List<ReportJob> listReports() {
        return storage.list();
    }

    enum ReportType {
        SALES_SUMMARY,
        INVENTORY_SNAPSHOT
    }
}
