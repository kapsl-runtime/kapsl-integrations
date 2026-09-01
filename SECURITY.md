# Security policy

Report suspected vulnerabilities privately through GitHub's **Security** tab by
opening a private vulnerability report. Do not include exploit details in a
public issue.

## Dependency policy

CI validates integration tests and release archives, audits workflow structure,
scans repository history for secrets, and runs CodeQL. Rust and Python
dependency audits are added with each integration's lockfile; known
vulnerabilities, unsound dependencies, yanked releases, and new maintenance
warnings fail the build unless a time-bounded exception is documented here.

No dependency-audit exceptions are currently approved for this repository.

## Release credentials

Release workflows use GitHub environments and OIDC trusted publishing. Do not
commit registry credentials, cloud service-account keys, GitHub App private
keys, or model-access tokens. Real GPU conformance may create only labeled,
ephemeral resources and must verify their teardown before a stable release can
publish.
