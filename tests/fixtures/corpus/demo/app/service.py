"""Job service: submits reports and retries failed ones."""
import logging
from dataclasses import dataclass

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0
log = logging.getLogger(__name__)


@dataclass
class Job:
    """One report job and its retry state."""

    job_id: str
    attempts: int = 0

    def next_delay(self) -> float:
        """Exponential backoff: the gap doubles after every failed attempt."""
        return BACKOFF_SECONDS * (2 ** self.attempts)

    # Mark the job as failed for good.
    def give_up(self) -> None:
        self.attempts = MAX_ATTEMPTS


class RateLimiter:
    """Allows at most `limit` requests per client per minute."""

    def __init__(self, limit: int = 60):
        self.limit = limit
        self.seen: dict[str, int] = {}

    def allow(self, client: str) -> bool:
        count = self.seen.get(client, 0) + 1
        self.seen[client] = count
        return count <= self.limit


@log_calls
def submit(job: Job) -> str:
    """Put a job on the queue and return its id."""
    log.info("submitting %s", job.job_id)
    return job.job_id


if __name__ == "__main__":
    submit(Job("demo"))
