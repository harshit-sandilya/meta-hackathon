from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------
class FileTreeEntry(BaseModel):
    path: str = Field(..., description="Relative path from sandbox root")
    is_dir: bool = Field(False)
    size_bytes: int = Field(0)
    last_modified: str | None = Field(None)


class FilePatch(BaseModel):
    path: str
    unified_diff: str | None = None
    new_content: str | None = None

    @model_validator(mode="after")
    def exactly_one_change(self) -> "FilePatch":
        if self.unified_diff is None and self.new_content is None:
            raise ValueError("FilePatch must supply either unified_diff or new_content")
        if self.unified_diff is not None and self.new_content is not None:
            raise ValueError(
                "FilePatch must supply unified_diff OR new_content, not both"
            )
        return self
