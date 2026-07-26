# Release checklist

The project is preparing its initial `0.1.0` release at Alpha maturity and
must not be published until the owner resolves the remaining decisions below.

## One-time publication decisions

- [x] License the project under MIT and add matching package and citation
      metadata.
- [x] Set the distribution name to `qualitative-qsim`.
- [ ] Claim or create `qualitative-qsim` on PyPI.
- [x] Configure PyPI Trusted Publishing for repository
      `alanoursland/qualitative-qsim`, workflow
      `publish-pypi.yml`, and GitHub environment `pypi`.
- [ ] Create the `pypi` GitHub environment and add an environment protection
      rule requiring approval before publication.
- [ ] Add author/maintainer metadata approved by the owner.
- [x] Confirm the repository, issue tracker, and publication URLs.

## Qualification

- [ ] Install the development environment from `requirements-dev.txt`, then
      install the repository with `--no-build-isolation -e .`.
- [x] Run the symbolic/reference CI matrix on every supported Python version.
- [x] Run `python -m pytest` with the required Torch dependency on CPU.
- [ ] On a CUDA qualification machine, run `python -m pytest tests/test_tensor.py`
      and `python benchmarks/bench_tensor.py`. Attach the output to the release
      record; it includes device/runtime details, synchronized raw samples,
      memory state, and transfer policy.
- [x] Run `python docs/tutorial/make_figures.py` and confirm
      `git diff --exit-code -- docs/tutorial/figures`.
- [ ] Run `python -m build`, install both the wheel and source distribution in
      clean environments, and smoke-test `import qrlib`.
- [x] Verify the CI matrix is green.

## Release

- [x] Move notable entries from `CHANGELOG.md`'s Unreleased section into a
      heading for the release version and date.
- [x] Confirm `0.1.0` as the intended final PEP 440 version for this alpha
      maturity release.
- [ ] Commit the version and changelog updates.
- [ ] Create an annotated `v<version>` tag from the qualified commit.
- [ ] Build artifacts from that tag and inspect wheel/sdist contents and
      metadata.
- [ ] Optionally publish the candidate to TestPyPI and smoke-test it.
- [ ] Publish the GitHub release with the changelog and qualification record;
      `.github/workflows/publish-pypi.yml` will rebuild and check the
      distributions, then request approval from the `pypi` environment before
      publishing them through Trusted Publishing.

## After publication

- [ ] Verify installation from PyPI in a clean environment and smoke-test
      both reference and tensor execution.
- [ ] Start a new Unreleased section in `CHANGELOG.md`.
- [ ] Review README, architecture, roadmap, open questions, tutorial output,
      benchmark claims, and package metadata for status drift.
