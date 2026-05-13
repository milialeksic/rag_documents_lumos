import os
from dotenv import load_dotenv
from notion_client import Client

load_dotenv()

NOTION_TOKEN = os.getenv("NOTION_TOKEN")
OUTPUT_PATH = "data/notion/"

def get_block_text(client, block_id):
    """Recursively extract text from blocks."""
    try:
        blocks = client.blocks.children.list(block_id=block_id)
        text_parts = []

        for block in blocks["results"]:
            block_type = block["type"]

            if block_type in [
                "paragraph", "heading_1", "heading_2", "heading_3",
                "bulleted_list_item", "numbered_list_item",
                "quote", "callout", "toggle", "to_do"
            ]:
                rich_text = block[block_type].get("rich_text", [])
                text = "".join([t["plain_text"] for t in rich_text])
                if text.strip():
                    if block_type == "heading_1":
                        text_parts.append(f"# {text}")
                    elif block_type == "heading_2":
                        text_parts.append(f"## {text}")
                    elif block_type == "heading_3":
                        text_parts.append(f"### {text}")
                    elif block_type == "bulleted_list_item":
                        text_parts.append(f"- {text}")
                    elif block_type == "numbered_list_item":
                        text_parts.append(f"1. {text}")
                    else:
                        text_parts.append(text)

            if block.get("has_children"):
                child_text = get_block_text(client, block["id"])
                if child_text:
                    text_parts.append(child_text)

        return "\n".join(text_parts)

    except Exception as e:
        print(f"  Error reading blocks: {e}")
        return ""

def get_page_title(page):
    """Extract title from page properties."""
    try:
        props = page.get("properties", {})
        for prop in props.values():
            if prop["type"] == "title":
                return "".join([t["plain_text"] for t in prop["title"]])
    except:
        pass
    return "Untitled"

def fetch_all_pages(client):
    """Get all pages accessible with this token."""
    all_pages = []
    cursor = None

    while True:
        params = {
            "filter": {"value": "page", "property": "object"},
            "page_size": 100
        }
        if cursor:
            params["start_cursor"] = cursor

        response = client.search(**params)
        all_pages.extend(response["results"])

        if not response["has_more"]:
            break
        cursor = response["next_cursor"]

    return all_pages

def export_notion_pages():
    client = Client(auth=NOTION_TOKEN)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    print("Searching for accessible Notion pages...")
    pages = fetch_all_pages(client)
    print(f"Found {len(pages)} pages")

    saved = 0
    skipped = 0

    for i, page in enumerate(pages):
        page_id = page["id"]
        title = get_page_title(page)
        safe_title = title.replace("/", "-").replace("\\", "-").replace(":", "-")[:80]

        print(f"  [{i+1}/{len(pages)}] {title[:60]}")

        text = get_block_text(client, page_id)

        if text.strip():
            filepath = os.path.join(OUTPUT_PATH, f"{safe_title}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n{text}")
            saved += 1
        else:
            skipped += 1
            print(f"    → empty, skipping")

    print(f"\nDone! Saved {saved} pages, skipped {skipped} empty pages")
    print(f"Files saved to: {OUTPUT_PATH}")
    print(f"Now run: python ingest.py")

if __name__ == "__main__":
    export_notion_pages()