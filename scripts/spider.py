#!/usr/bin/env python3
import logging
import re
import shutil
import subprocess
import time
import traceback
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union
from urllib.parse import urljoin

import git
import requests
import unidiff
import yaml
from bs4 import BeautifulSoup
from bs4.element import Tag

VULN_DIR = Path(__file__).parent.parent / "vuln"
LINUX_REPO = git.Repo(Path(__file__).parent.parent / "linux")

logging.basicConfig(level=logging.INFO, format="%(message)s")


class SyzkallerBugSpider:
    base_dir: Path
    session: requests.Session
    cache_dir: Path

    def __init__(self, base_dir: Union[str, Path] = "vuln") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; SyzkallerBugSpider/1.0)"})

        self.cache_dir = self.base_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)

    def fetch_page(self, url: str) -> str:
        url_hash: str = md5(url.encode("utf-8")).hexdigest()
        cache_file: Path = self.cache_dir / f"{url_hash}.html"
        while True:
            try:
                logging.info(f"📥 Fetching: {url}")

                if cache_file.exists():
                    text = cache_file.read_text()
                else:
                    response: requests.Response = self.session.get(url, timeout=30)
                    response.raise_for_status()
                    text = response.text

                assert "Too many requests" not in text, "Rate limit exceeded"

                if not cache_file.exists():
                    cache_file.write_text(text)

                return text
            except (requests.RequestException, AssertionError) as e:
                logging.error(f"⚠️ Error fetching {url}: {e}")

                cache_file.unlink(missing_ok=True)
                time.sleep(30)

    def extract_bugs_from_page(self, html_content: str, base_url: str) -> List[Dict[str, str]]:
        if not html_content:
            return []

        bugs: List[Dict[str, str]] = []
        soup: BeautifulSoup = BeautifulSoup(html_content, "html.parser")

        bug_links = soup.find_all("a", href=re.compile(r"/bug\?extid=([a-f0-9]+)"))

        for link in bug_links:
            if not isinstance(link, Tag):
                continue
            href = link.get("href")
            if isinstance(href, list):
                href = href[0] if href else None
            extid_match = re.search(r"extid=([a-f0-9]+)", href) if href else None

            if extid_match:
                extid: str = extid_match.group(1)
                bug_url: str = urljoin(base_url, href)
                bug_title: str = link.get_text(strip=True)

                bugs.append(
                    {
                        "extid": extid,
                        "bug_link": bug_url,
                        "title": bug_title,
                        "source_page": base_url,
                    }
                )

        return bugs

    def create_bug_folder(self, bug_info: Dict[str, str], repo: git.Repo) -> Path:
        extid: str = bug_info["extid"]
        folder_path: Path = self.base_dir / extid

        folder_path.mkdir(exist_ok=True)
        config_data: Dict[str, int | str | List[str]] = {
            "id": extid,
            "bug_link": bug_info["bug_link"],
            "title": bug_info["title"],
            "source_page": bug_info["source_page"],
        }

        config_file: Path = folder_path / "config.yaml"
        soup: BeautifulSoup = BeautifulSoup(self.fetch_page(bug_info["bug_link"]), "html.parser")

        trigger_commit, fix_commit = self.save_patch(
            repo,
            soup,
            folder_path / "patch.diff",
        )
        self.save_sanitizer_report(soup, folder_path / "report.txt")

        config_data["trigger_commit"] = trigger_commit
        config_data["fix_commit"] = fix_commit
        config_data["datetime"] = repo.commit(fix_commit).committed_datetime.isoformat()

        commit_message = repo.commit(fix_commit).message
        if isinstance(commit_message, memoryview):
            commit_message = commit_message.tobytes().decode("utf-8", errors="replace")
        elif isinstance(commit_message, (bytes, bytearray)):
            commit_message = commit_message.decode("utf-8", errors="replace")
        else:
            commit_message = str(commit_message)
        config_data["fix_commit_message"] = commit_message

        config_data["submodule"] = self.fetch_submodule(folder_path / "patch.diff")

        config_data["hunk_count"], config_data["covered_count"] = self.fetch_root_cause_cover(
            folder_path / "patch.diff",
            folder_path / "report.txt",
        )

        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f, sort_keys=False, allow_unicode=True)

        logging.info(f"📁 Created folder: {folder_path}")
        return folder_path

    def fetch_submodule(self, patch_file: Path) -> List[str]:
        assert patch_file.exists(), "patch file does not exist"

        submodule: Set[str] = set()
        for file in unidiff.PatchSet(patch_file.read_text()):
            module_path = Path(file.path).parent

            parts = module_path.parts
            if len(parts) == 0:
                continue

            if (parts[0] in ["fs", "net"]) and len(parts) > 2:
                module_path = Path(parts[0]) / parts[1]

            submodule.add(module_path.as_posix())

        return sorted(list(submodule))

    def fetch_root_cause_cover(self, patch_file: Path, report_txt: Path) -> Tuple[int, int]:
        patch = unidiff.PatchSet(patch_file.read_text())

        hunk_count = sum(len(file) for file in patch)
        report_content = report_txt.read_text()

        covered_count = 0
        for file in patch:
            for hunk in file:
                start_line = max(hunk.source_start - 50, 0)
                end_line = hunk.source_start + hunk.source_length + 50
                if any(f"{file.path}:{line}" in report_content for line in range(start_line, end_line)):
                    covered_count += 1

        return hunk_count, covered_count

    def save_sanitizer_report(self, soup: BeautifulSoup, report_file: Path) -> None:
        if report_file.exists():
            logging.info(f"📝 Report already exists: {report_file}")
            return

        logging.info(f"📝 Saving sanitizer report to: {report_file}")
        crash_info = soup.find("pre")
        assert crash_info, "No crash information found in the bug page"
        report_file.write_text(crash_info.get_text())

    def save_patch(
        self,
        repo: git.Repo,
        soup: BeautifulSoup,
        patch_file: Path,
    ) -> Tuple[str, str]:
        logging.info(f"📄 Saving patch to: {patch_file}")

        fix_commit_text = soup.find(string=re.compile(r"Fix commit:"))
        assert fix_commit_text, "No fix commit information found in the bug page"
        parent = fix_commit_text.parent
        assert parent, "Parent element not found for fix commit text"
        mono_span = parent.find_next("span", class_="mono")
        assert mono_span, "No span with class 'mono' found after fix commit text"
        commit_hash_text = mono_span.get_text().strip().split("\n")[0]
        assert all(c in "0123456789abcdef" for c in commit_hash_text), "Invalid commit hash format"

        fix_commit = repo.commit(commit_hash_text).hexsha
        parent_commit = repo.commit(fix_commit).parents[0].hexsha

        if not patch_file.exists():
            with patch_file.open("wb") as f:
                subprocess.run(
                    ["git", "diff", parent_commit, fix_commit],
                    cwd=repo.working_tree_dir,
                    stdin=subprocess.DEVNULL,
                    stdout=f,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )

        return parent_commit, fix_commit

    def process_fixed_bugs_page(self, url: str, repo: git.Repo) -> List[Path]:
        """Process a single fixed bugs page."""
        logging.info(f"\n📄 Processing page: {url}")

        html_content: str = self.fetch_page(url)
        if not html_content:
            logging.warning(f"⚠️ Failed to fetch page: {url}")
            return []

        bugs: List[Dict[str, str]] = self.extract_bugs_from_page(html_content, url)
        logging.info(f"🐞 Found {len(bugs)} bugs on this page")

        created_folders: List[Path] = []
        for bug_info in bugs:
            folder_path: Path = VULN_DIR / bug_info["extid"]
            try:
                self.create_bug_folder(bug_info, repo)
                created_folders.append(folder_path)
            except AssertionError as e:
                logging.error(f"⚠️ Error for bug {bug_info['extid']}: {e}")
                shutil.rmtree(folder_path)
                continue

        return created_folders

    def run(self, targets: List[Tuple[str, git.Repo]]) -> List[Path]:
        """Main method to process all URLs."""
        logging.info("🕸️ Starting Syzkaller Bug Spider...")
        logging.info(f"📂 Output directory: {self.base_dir.absolute()}")

        all_folders: List[Path] = []
        for url, repo in targets:
            try:
                folders: List[Path] = self.process_fixed_bugs_page(url, repo)
                all_folders.extend(folders)
            except Exception as e:
                logging.error(f"⚠️ Error processing {url}: {e}")
                traceback.print_exc()
                continue

        logging.info(f"\n✅ Completed! Created {len(all_folders)} bug folders.")

        return all_folders


def main() -> None:
    spider: SyzkallerBugSpider = SyzkallerBugSpider()
    spider.run([("https://syzkaller.appspot.com/upstream/fixed", LINUX_REPO)])


if __name__ == "__main__":
    main()
