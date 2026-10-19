# Skills

Claude skills, one folder per skill.

| Folder | Command | What it does |
| --- | --- | --- |
| `video-editing/` | `/setup-tiktok-edit` | Edits raw setup clips into a beat-synced vertical TikTok. |
| `grill-me/` | `/grill-me` | Interviews you relentlessly to sharpen a plan or design. Hands off to `grilling`. |
| `grilling/` | `/grilling` | The interview itself; `grill-me` needs it. |
| `teach/` | `/teach` | Teaches a topic over several sessions, keeping lessons and notes in the folder you run it from. |

`grill-me`, `grilling` and `teach` are from [mattpocock/skills](https://github.com/mattpocock/skills) (MIT, license in each folder).

`.claude/skills/setup-tiktok-edit/` is only a pointer to `video-editing/`, so cloud sessions on this repo (which load skills from `.claude/skills/`) find the skill. Edit `video-editing/`, not the pointer.
