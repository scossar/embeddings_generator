import frontmatter
import chromadb
from chromadb import Collection
import re
import unidecode
import sqlite3
import json

from postchunker import extract_sections
from pathlib import Path

postspath = "/home/scossar/zalgorithm/content"
datapath = "/home/scossar/zalgorithm/data/fragments/sections.json"
sqlite_path = "/home/scossar/projects/python/embeddings_generator/sqlite"

# NOTES ##########################################################################################################
# see https://docs.trychroma.com/docs/embeddings/embedding-functions for details about custom embedding functions,
# i.e, creating one for for "all-mpnet-base-v2"
# ################################################################################################################


class EmbeddingGenerator:
    def __init__(
        self,
        content_directory: str = "/home/scossar/zalgorithm/content",
        html_directory: str = "/home/scossar/zalgorithm/public",
        collection_name: str = "zalgorithm",
    ):
        self.skip_dirs: set[str] = {  # these are mostly wrong
            "node_modules",
            ".git",
            ".obsidian",
            "__pycache__",
            "venv",
            ".venv",
        }
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient()  # chroma will use the default `chroma` directory in the base of the project for persistence
        self.collection = self.get_or_create_collection()
        self.content_directory = content_directory
        self.html_directory = html_directory
        self.con = self.get_db_connection()
        self.create_sections_table(self.con)

    def get_db_connection(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"{sqlite_path}/sections.db")
        return con

    def create_sections_table(self, con: sqlite3.Connection) -> None:
        cur = con.cursor()
        cur.execute("""
CREATE TABLE IF NOT EXISTS sections (
    id INTEGER PRIMARY KEY,  -- Auto-incrementing rowid
    section_id TEXT NOT NULL UNIQUE,
    post_id TEXT NOT NULL,
    section_heading_slug TEXT NOT NULL,
    html_heading TEXT NOT NULL,
    html_fragment TEXT NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(post_id, section_heading_slug)
);
        """)
        cur.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_section_id ON sections(section_id);"
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_post_id ON sections(post_id);")

        return None

    def save_to_sqlite(
        self,
        section_id: str,
        post_id: str,
        section_heading_slug: str,
        html_heading: str,
        html_fragment: str,
        updated_at: float,
    ) -> int:
        cursor = self.con.execute(
            """
INSERT INTO sections 
    (section_id, post_id, section_heading_slug, html_heading, html_fragment, updated_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(section_id) DO UPDATE SET
    html_heading = excluded.html_heading,
    html_fragment = excluded.html_fragment,
    updated_at = excluded.updated_at
RETURNING id
        """,
            (
                section_id,
                post_id,
                section_heading_slug,
                html_heading,
                html_fragment,
                updated_at,
            ),
        )

        return cursor.fetchone()[0]

    def get_or_create_collection(self) -> Collection:
        return self.chroma_client.get_or_create_collection(name=self.collection_name)

    # TODO: this is kind of chaotic; remember it's passed markdown files, not html files.
    # either respect 'draft' frontmatter boolean, or a 'private' boolean;
    def _should_process_file(self, filepath: Path) -> bool:
        if any(part.startswith(".") for part in filepath.parts):
            return False
        if any(part.startswith("_") for part in filepath.parts):
            return False
        if any(skip_dir in filepath.parts for skip_dir in self.skip_dirs):
            return False
        if filepath.suffix.lower() not in (".md", ".markdown"):
            return False
        if filepath.name == "search.md":
            return False
        return True

    # Hoping this matches Hugo's implementation
    def _slugify(self, title: str) -> str:
        title = unidecode.unidecode(title).lower()
        title = re.sub(r"[^a-z0-9\s-]", "", title)
        title = re.sub(r"[\s_]+", "-", title)
        title = title.strip("-")  # strip leading/trailing hyphens
        return title

    def _is_up_to_date(self, file_id: str, file_mtime: float) -> bool:
        existing = self.collection.get(ids=file_id, limit=1)

        if not existing["ids"] or not existing["metadatas"]:
            return False

        last_updated_at = existing["metadatas"][0].get("updated_at", 0)

        if not isinstance(last_updated_at, (int, float)):
            return False  # if it's an invalid timestamp

        return (
            last_updated_at + 1.0 >= file_mtime
        )  # 1 second tolerance for rounding errors

    def generate_embeddings(self):
        """
        Generate embeddings for blog content
        """
        paths_data = []
        for path in Path(self.content_directory).rglob("*"):
            if not self._should_process_file(path):
                continue
            path_data = self.generate_embedding(path)
            paths_data.append(path_data)

        # data for the Hugo build process
        sections_map = {}
        for path in paths_data:
            for section in path:
                relative_path = section["relative_path"]
                db_id = section["db_id"]
                sections_map[relative_path] = {"db_id": db_id}

        with open(datapath, "w") as f:
            json.dump(sections_map, f)

    def get_file_paths(self, md_path: Path) -> tuple[str, str] | tuple[None, None]:
        try:
            rel_path = md_path.relative_to(self.content_directory)
        except ValueError:  # if md_path isn't a subpath of content_directory
            print(
                f"{md_path} isn't relative to the content directory ({self.content_directory})"
            )
            return None, None

        rel_path_parts = rel_path.with_suffix("").parts
        rel_path_parts = tuple(
            s.lower() for s in rel_path_parts
        )  # it's possible to end up with an uppercase md filename
        html_path = Path(self.html_directory) / Path(*rel_path_parts) / "index.html"

        if html_path.exists():
            return str(html_path), str(Path(*rel_path_parts))
        else:
            print(f"No file exists at {html_path}")
            return None, None

    def generate_embedding(self, filepath: Path) -> list[dict[str, str]] | None:
        html_path, relative_path = self.get_file_paths(filepath)
        if not html_path or not relative_path:
            return None

        print(f"Processing {relative_path}")

        post = frontmatter.load(str(filepath))
        file_mtime = filepath.stat().st_mtime
        title = str(post.get("title"))
        post_id = post.get("id", None)
        if not post_id:
            print(
                f"The post '{title}' is missing an 'id' field. Skipping generating an embedding."
            )
            return None

        sections = extract_sections(html_path, relative_path)

        sections_data = []
        for section in sections:
            html_fragment = section["html_fragment"]
            html_heading = section["html_heading"]
            page_heading = section["headings_path"][0]
            section_heading = section["headings_path"][-1]
            section_heading_id = section["heading_id"]
            embeddings_text = section["embeddings_text"]
            section_id = f"{post_id}-{section_heading_id}"

            db_id = self.save_to_sqlite(
                section_id=section_id,
                post_id=str(post_id),  # it's a string!
                section_heading_slug=section_heading_id,
                html_heading=html_heading,
                html_fragment=html_fragment,
                updated_at=file_mtime,
            )
            self.con.commit()  # do I also need to be closing the connection? (no?)

            section_data = {
                "heading_id": section_heading_id,  # might not need this?
                "relative_path": section["heading_href"],
                "db_id": db_id,
            }
            sections_data.append(section_data)

            for index, text in enumerate(embeddings_text):
                embedding_id = f"{post_id}-{index}-{section_heading_id}"

                metadatas = {
                    "page_title": page_heading,
                    "section_heading": section_heading,
                    "db_id": db_id,
                    "updated_at": file_mtime,
                }

                self.collection.upsert(
                    ids=embedding_id, metadatas=metadatas, documents=text
                )

        return sections_data

    # for testing
    def query_collection(self, query: str):
        results = self.collection.query(
            query_texts=[query],
            n_results=7,
            include=["metadatas", "documents", "distances"],
        )

        if not (results["metadatas"] and results["documents"] and results["distances"]):
            return

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        zipped = zip(ids, documents, metadatas, distances)

        for _, document, metadata, distance in zipped:
            print("\n", metadata)
            print(distance, "\n")


embeddings_generator = EmbeddingGenerator()
embeddings_generator.generate_embeddings()
