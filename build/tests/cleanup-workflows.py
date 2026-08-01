#!/usr/bin/env python3

import re
import subprocess
import tempfile
from pathlib import Path

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_DIRECTORY = REPOSITORY_ROOT / ".github" / "workflows"
DELETE_OLD_WORKFLOW = WORKFLOW_DIRECTORY / "delete-old-images.yml"
DELETE_UNTAGGED_WORKFLOW = WORKFLOW_DIRECTORY / "delete-untagged-images.yml"
EXPECTED_CONCURRENCY_GROUP = (
    "package-cleanup-${{ github.repository_owner }}-${{ github.event.repository.name }}"
)


class GitHubWorkflowLoader(yaml.SafeLoader):
    pass


GitHubWorkflowLoader.yaml_implicit_resolvers = {
    first_character: [
        (tag, pattern)
        for tag, pattern in resolvers
        if tag != "tag:yaml.org,2002:bool"
    ]
    for first_character, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
GitHubWorkflowLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|false)$", re.IGNORECASE),
    list("tTfF"),
)


def load_workflow(path: Path) -> dict:
    with path.open(encoding="utf-8") as workflow_file:
        workflow = yaml.load(workflow_file, Loader=GitHubWorkflowLoader)

    assert isinstance(workflow, dict), f"{path} must contain a YAML mapping"
    assert "on" in workflow, f"{path} did not preserve its on key"
    assert True not in workflow, f"{path} parsed on as a YAML 1.1 boolean"
    return workflow


def cleanup_job(workflow: dict) -> dict:
    jobs = workflow["jobs"]
    assert len(jobs) == 1
    return next(iter(jobs.values()))


def github_script_step(workflow: dict) -> dict:
    steps = cleanup_job(workflow)["steps"]
    matching_steps = [step for step in steps if step.get("uses") == "actions/github-script@v7"]
    assert len(matching_steps) == 1
    return matching_steps[0]


def assert_common_cleanup_behavior(path: Path, workflow: dict) -> str:
    assert workflow["permissions"] == {"contents": "read", "packages": "write"}
    assert workflow["concurrency"] == {
        "group": EXPECTED_CONCURRENCY_GROUP,
        "cancel-in-progress": False,
    }

    step = github_script_step(workflow)
    assert step["with"]["github-token"] == "${{ secrets.GITHUB_TOKEN }}"
    assert step["env"]["PACKAGE_OWNER"] == "${{ github.repository_owner }}"
    assert step["env"]["PACKAGE_NAME"] == "${{ github.event.repository.name }}"

    script = step["with"]["script"]
    required_fragments = (
        "github.paginate",
        "/users/${encodedOwner}/packages/container/${encodedPackageName}/versions",
        "encodeURIComponent(owner)",
        "encodeURIComponent(packageName)",
        "status === 404",
        "Candidate version id=",
        "updated_at=",
        "if (dryRun)",
        "github.request(`DELETE ${deletePath}`)",
    )
    for fragment in required_fragments:
        assert fragment in script, f"{path} script is missing {fragment!r}"

    delete_request_index = script.index("github.request(`DELETE ${deletePath}`)")
    assert script.index("github.paginate") < delete_request_index
    assert script.index("if (dryRun)") < delete_request_index

    for forbidden_fragment in ("DELETE_PACKAGES_TOKEN", "orgs/", "github-script@v6"):
        assert forbidden_fragment not in path.read_text(encoding="utf-8")

    return script


def assert_javascript_syntax(path: Path, script: str) -> None:
    wrapped_script = f"async function validateEmbeddedScript() {{\n{script}\n}}\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mjs", encoding="utf-8") as script_file:
        script_file.write(wrapped_script)
        script_file.flush()
        subprocess.run(
            ["node", "--check", script_file.name],
            check=True,
            text=True,
            capture_output=True,
        )


def assert_delete_old_behavior(workflow: dict, script: str) -> None:
    triggers = workflow["on"]
    assert set(triggers) == {"workflow_dispatch"}
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert inputs["age"]["default"] == "1 Month"
    assert inputs["dry_run"]["type"] == "boolean"
    assert inputs["dry_run"]["default"] is True
    assert "All Packages" in inputs["age"]["options"]

    assert 'new Set(["latest", "latest-cuda"])' in script
    assert 'protectedReasons.push("newest version by updated_at")' in script
    assert "nowMilliseconds - version.updatedAtMilliseconds" in script
    assert "ageMilliseconds <= maximumAgeMilliseconds" in script
    assert "Unsupported age selection" in script
    assert "version.tags.length === 0" not in script


def assert_delete_untagged_behavior(workflow: dict, script: str) -> None:
    triggers = workflow["on"]
    assert set(triggers) == {"workflow_dispatch", "workflow_run"}
    assert triggers["workflow_run"]["workflows"] == ["Docker Build"]
    assert triggers["workflow_run"]["types"] == ["completed"]
    dry_run_input = triggers["workflow_dispatch"]["inputs"]["dry_run"]
    assert dry_run_input["type"] == "boolean"
    assert dry_run_input["default"] is True

    job_condition = cleanup_job(workflow)["if"]
    assert "workflow_dispatch" in job_condition
    assert "workflow_run.conclusion == 'success'" in job_condition
    assert 'const dryRun = isManualRun ? parseBooleanInput("DRY_RUN") : false;' in script
    assert "version.tags.length === 0" in script


def main() -> None:
    for workflow_path in sorted(WORKFLOW_DIRECTORY.glob("*.y*ml")):
        load_workflow(workflow_path)

    old_workflow = load_workflow(DELETE_OLD_WORKFLOW)
    old_script = assert_common_cleanup_behavior(DELETE_OLD_WORKFLOW, old_workflow)
    assert_delete_old_behavior(old_workflow, old_script)
    assert_javascript_syntax(DELETE_OLD_WORKFLOW, old_script)

    untagged_workflow = load_workflow(DELETE_UNTAGGED_WORKFLOW)
    untagged_script = assert_common_cleanup_behavior(
        DELETE_UNTAGGED_WORKFLOW, untagged_workflow
    )
    assert_delete_untagged_behavior(untagged_workflow, untagged_script)
    assert_javascript_syntax(DELETE_UNTAGGED_WORKFLOW, untagged_script)

    print("Cleanup workflow checks passed.")


if __name__ == "__main__":
    main()
