import nox
import tempfile

py_versions = ["3.10", "3.11"]
folders = ["src", "tests"]


@nox.session(python=py_versions)
def tests(session):
    """Run the test suite."""
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov")


@nox.session(python=py_versions)
def lint(session):
    """Lint using flake8."""
    args = session.posargs or folders
    session.install(
        "flake8",
        "flake8-black",
        "flake8-bandit",
        "flake8-annotations",
        "flake8-docstrings"
    )
    session.run("flake8", *args)


@nox.session(python=py_versions)
def black(session):
    """Run black code formatter."""
    args = session.posargs or folders
    session.install("black")
    session.run("black", *args)


@nox.session(python=py_versions)
def safety(session):
    """Scan dependencies for insecure packages."""
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--with",
            "dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True
        )
        session.install("safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")


@nox.session(python=py_versions)
def mypy(session):
    """Static type-check using mypy."""
    args = session.posargs or folders
    session.install("mypy")
    session.run("mypy", *args)


@nox.session(python=py_versions)
def pytype(session):
    """Static type-check using pytype."""
    args = session.posargs or ["--disable=import-error", *folders]
    session.install("pytype")
    session.run("pytype", *args)


"""
#TODO make auto documentation
@nox.session(python=py_versions)
def docs(session) -> None:
    session.install("sphinx", "sphinx_autodoc_typehints")
    session.run(
        "poetry",
        "run",
        "sphinx-build", "docs", "docs/_build", external=True)
"""
