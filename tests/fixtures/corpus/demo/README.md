# Demo service

A small service that produces reports and retries failed jobs.

## Running it

Start the API with `python -m demo`.

### Local AWS

Point the service at LocalStack by setting `AWS_ENDPOINT_URL`.

## Fault tolerance

### Retries

Failed jobs are retried with an exponential backoff, at most three times.

### Rate limiting

Each client may make sixty requests per minute.
