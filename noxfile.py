import nox
import tempfile

py_versions = ["3.10"]
folders = ["src", "tests"]


@nox.session(python=py_versions)
def tests(session):
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov")


@nox.session(python=py_versions)
def lint(session):
    args = session.posargs or folders
    session.install(
        "flake8",
        "flake8-black",
        "flake8-bandit"
    )
    session.run("flake8", *args)


@nox.session(python=py_versions)
def black(session):
    args = session.posargs or folders
    session.install("black")
    session.run("black", *args)


@nox.session(python=py_versions)
def safety(session):
    with tempfile.NamedTemporaryFile() as requirements:
        session.run(
            "poetry",
            "export",
            "--dev",
            "--format=requirements.txt",
            "--without-hashes",
            f"--output={requirements.name}",
            external=True
        )
        session.install("safety")
        session.run("safety", "check", f"--file={requirements.name}", "--full-report")
