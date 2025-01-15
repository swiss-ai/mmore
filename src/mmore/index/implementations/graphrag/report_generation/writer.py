from io import StringIO

from .utils import CommunityReportResult


class CommunityReportWriter:
    def write(self, report: CommunityReportResult) -> str:
        fp = StringIO()

        try:
            fp.write(f"# {report.title}\n")
            fp.write(f"{report.summary}\n")

            for finding in report.findings:
                fp.write(f"## {finding.summary}\n")
                fp.write(f"{finding.explanation}\n")

            return fp.getvalue()
        finally:
            fp.close()
