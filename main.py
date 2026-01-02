import frontmatter
import chromadb
from chromadb import Collection
import re
import unidecode

from postchunker import extract_sections
from pathlib import Path

postspath = "/home/scossar/zalgorithm/content"

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
        for path in Path(self.content_directory).rglob("*"):
            if not self._should_process_file(path):
                continue
            self.generate_embedding(path)

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

    def generate_embedding(self, filepath: Path) -> None:
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

        for section in sections:
            html_fragment = section["html_fragment"]
            html_heading = section["html_heading"]
            page_heading = section["headings_path"][0]
            section_heading = section["headings_path"][-1]
            section_heading_slug = self._slugify(section_heading)
            embeddings_text = section["embeddings_text"]

            for index, text in enumerate(embeddings_text):
                embedding_id = f"{post_id}-{index}-{section_heading_slug}"
                # TODO:  (maybe) uncomment after testing
                # if self._is_up_to_date(embedding_id, file_mtime):
                #     print(f"Not indexing {title}. Up to date.")
                #     return None

                metadatas = {
                    "page_title": page_heading,
                    "section_heading": section_heading,
                    "html_heading": html_heading,
                    "html_fragment": html_fragment,
                    "updated_at": file_mtime,
                }

                self.collection.upsert(
                    ids=embedding_id, metadatas=metadatas, documents=text
                )

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


# test_path = "/home/scossar/zalgorithm/content/notes/a-simple-document-for-testing.md"
# test_path = "/home/scossar/zalgorithm/content/notes/roger-bacon-as-magician.md"
# test_path = "/home/scossar/zalgorithm/content/notes/notes-on-cognitive-and-morphological-patterns.md"
embeddings_generator = EmbeddingGenerator()
# embeddings_generator.generate_embedding(Path(test_path))
embeddings_generator.generate_embeddings()
# embeddings_generator.query_collection("How do I stop tracking a file with git?")
