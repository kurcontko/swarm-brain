# Vendored agent skills

`skills/` holds a read-only copy of 34 skills from the public
[`cockroachlabs/cockroachdb-skills`](https://github.com/cockroachlabs/cockroachdb-skills)
repository. They are CockroachDB Labs' work, not this project's, and they are
reproduced here only so an agent session can use them offline and so the exact
revision in use is auditable.

Every file is pinned by content hash in [`../skills-lock.json`](../skills-lock.json),
which also records each skill's upstream path. Nothing here is modified locally;
to change a skill, update it upstream and re-vendor.

Use of these files is governed by the upstream repository's license, not by this
project's [MIT license](../LICENSE).
