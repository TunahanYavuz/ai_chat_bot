use std::collections::VecDeque;
use std::path::{Path, PathBuf};

const MAX_DB_DISCOVERY_DEPTH: usize = 10;

pub fn find_first_db_file(root: &Path) -> Option<PathBuf> {
    let mut queue: VecDeque<(PathBuf, usize)> = VecDeque::from([(root.to_path_buf(), 0)]);
    while let Some((dir, depth)) = queue.pop_front() {
        let read_dir = std::fs::read_dir(&dir).ok()?;
        for entry in read_dir.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if depth < MAX_DB_DISCOVERY_DEPTH {
                    queue.push_back((path, depth + 1));
                }
                continue;
            }
            if path
                .extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("db"))
                .unwrap_or(false)
            {
                return Some(path);
            }
        }
    }
    None
}
