"""
File Permissions Utility

Kalıcı permission çözümü için utility fonksiyonlar:
- Shared group (bist-pattern) kullanımı
- Otomatik chown/chmod
- Group-writable permissions (775)
"""
import os
import stat
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Shared group name
SHARED_GROUP = 'bist-pattern'
WEB_USER = 'www-data'
WEB_GROUP = 'www-data'

# Default permissions: 775 (rwxrwxr-x) for directories, 664 (rw-rw-r--) for files
DIR_PERMISSIONS = 0o775
FILE_PERMISSIONS = 0o664


def ensure_file_permissions(file_path: Path, owner: Optional[str] = None, group: Optional[str] = None, 
                           file_mode: Optional[int] = None, dir_mode: Optional[int] = None) -> bool:
    """
    Ensure file/directory has correct permissions for shared access.
    
    Args:
        file_path: Path to file or directory
        owner: Owner user (default: current user, or www-data if running as root)
        group: Group (default: bist-pattern if exists, else www-data)
        file_mode: File permissions (default: 664)
        dir_mode: Directory permissions (default: 775)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import pwd
        import grp
        
        # Determine owner
        if owner is None:
            current_uid = os.getuid()
            if current_uid == 0:  # Running as root
                owner = WEB_USER
            else:
                owner = pwd.getpwuid(current_uid).pw_name
        
        # Determine group
        if group is None:
            try:
                # Try to use shared group first
                grp.getgrnam(SHARED_GROUP)
                group = SHARED_GROUP
            except KeyError:
                # Fallback to www-data group
                group = WEB_GROUP
        
        # Set permissions
        if file_mode is None:
            file_mode = FILE_PERMISSIONS
        if dir_mode is None:
            dir_mode = DIR_PERMISSIONS
        
        # Ensure parent directory exists and has correct permissions
        parent = file_path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
            _set_permissions(parent, owner, group, dir_mode, is_dir=True)
        
        # Set permissions for the file/directory itself
        is_dir = file_path.is_dir() if file_path.exists() else False
        _set_permissions(file_path, owner, group, dir_mode if is_dir else file_mode, is_dir=is_dir)
        
        return True
    except Exception as e:
        logger.debug(f"⚠️ Could not set permissions for {file_path}: {e}")
        return False


def _set_permissions(path: Path, owner: str, group: str, mode: int, is_dir: bool = False) -> None:
    """Set ownership and permissions for a path"""
    try:
        import pwd
        import grp
        
        # Get UID and GID
        try:
            uid = pwd.getpwnam(owner).pw_uid
        except KeyError:
            uid = os.getuid()  # Fallback to current user
        
        try:
            gid = grp.getgrnam(group).gr_gid
        except KeyError:
            gid = os.getgid()  # Fallback to current group
        
        # Set ownership
        os.chown(path, uid, gid)
        
        # Set permissions
        os.chmod(path, mode)
        
    except PermissionError:
        # If we can't change ownership (e.g., not running as root), just set permissions
        try:
            os.chmod(path, mode)
        except Exception as e:
            logger.debug(f"Failed to chmod {path}: {e}")
    except Exception as e:
        logger.debug(f"Failed to set permissions for {path}: {e}")


def ensure_directory_permissions(dir_path: Path, owner: Optional[str] = None, 
                                group: Optional[str] = None, recursive: bool = False) -> bool:
    """
    Ensure directory has correct permissions (775) for shared access.
    
    Args:
        dir_path: Path to directory
        owner: Owner user (default: current user, or www-data if running as root)
        group: Group (default: bist-pattern if exists, else www-data)
        recursive: Apply recursively to all subdirectories
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        ensure_file_permissions(dir_path, owner=owner, group=group, dir_mode=DIR_PERMISSIONS)
        
        if recursive:
            for root, dirs, files in os.walk(dir_path):
                root_path = Path(root)
                ensure_file_permissions(root_path, owner=owner, group=group, dir_mode=DIR_PERMISSIONS)
                for d in dirs:
                    ensure_file_permissions(root_path / d, owner=owner, group=group, dir_mode=DIR_PERMISSIONS)
                for f in files:
                    ensure_file_permissions(root_path / f, owner=owner, group=group, file_mode=FILE_PERMISSIONS)
        
        return True
    except Exception as e:
        logger.debug(f"⚠️ Could not set directory permissions for {dir_path}: {e}")
        return False


def fix_existing_file_permissions(file_path: Path, owner: Optional[str] = None, 
                                  group: Optional[str] = None) -> bool:
    """
    Fix permissions for an existing file (useful for files created before this fix).
    
    Args:
        file_path: Path to file
        owner: Owner user (default: www-data)
        group: Group (default: bist-pattern if exists, else www-data)
    
    Returns:
        True if successful, False otherwise
    """
    if not file_path.exists():
        return False
    
    if owner is None:
        owner = WEB_USER
    
    if group is None:
        try:
            import grp
            grp.getgrnam(SHARED_GROUP)
            group = SHARED_GROUP
        except (KeyError, ImportError):
            group = WEB_GROUP
    
    mode = DIR_PERMISSIONS if file_path.is_dir() else FILE_PERMISSIONS
    return ensure_file_permissions(file_path, owner=owner, group=group, 
                                  dir_mode=mode if file_path.is_dir() else None,
                                  file_mode=mode if not file_path.is_dir() else None)

