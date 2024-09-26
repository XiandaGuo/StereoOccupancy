#pragma once

#include <algorithm>
#include <boost/filesystem.hpp>
#include <fstream>
#include <list>
#include <string>
#include <vector>


namespace FileUtils {

#define JOIN_PATH(root, path) \
    (root.back() == '/' ? root + path : root + '/' + path)

class FileUtil {
 public:
    FileUtil() = default;
    ~FileUtil() = default;

    // @brief get basename for a path
    static std::string Basename(const std::string &path) {
        std::string fpath = path;
        if (fpath.back() == '/') fpath.pop_back();
        boost::filesystem::path p(fpath);
        return p.filename().string();
    }

    // @brief get filename for a path
    static std::string Filename(const std::string &path) {
        boost::filesystem::path p(path);
        return p.stem().string();
    }

    static std::string ParentDir(const std::string &path) {
        boost::filesystem::path p(path);
        return p.parent_path().string();
    }

    // @brief get extension for a path
    static std::string Extension(const std::string &path) {
        boost::filesystem::path p(path);
        return p.extension().string();
    }

    // @brief determine if the file or directory exists
    static bool IsExists(const std::string &path) {
        boost::filesystem::path p(path);
        return boost::filesystem::exists(p);
    }

    // @brief determine if the path is a directory
    static bool IsDirectory(const std::string &path) {
        boost::filesystem::path p(path);
        return boost::filesystem::is_directory(p);
    }

    // @brief determine if the path is a regular file
    static bool IsRegularFile(const std::string &path) {
        boost::filesystem::path p(path);
        return boost::filesystem::is_regular_file(p);
    }

    // @brief create file
    static bool CreateFile(const std::string &path) {
        std::ofstream ofs(path, std::ios::trunc | std::ios::out);
        bool ret = ofs.is_open();
        ofs.close();
        return ret;
    }

    static bool CreateSymlinkFile(const std::string &target_path,
                                  const std::string &link_path) {
        boost::filesystem::path target_p(target_path);
        boost::filesystem::path link_p(link_path);
        try {
            boost::filesystem::create_symlink(target_p, link_p);
            return true;
        } catch (const boost::filesystem::filesystem_error &e) {
            return false;
        }
    }

    // @brief create directory
    static bool CreateDirectory(const std::string &path) {
        boost::filesystem::path p(path);
        return boost::filesystem::create_directory(p);
    }

    // @brief create directories
    static bool CreateDirectories(const std::string &path) {
        boost::filesystem::path p(path);
        return boost::filesystem::create_directories(p);
    }

    // @brief delete file or directory
    static bool DeleteFile(const std::string &path) {
        boost::filesystem::path p(path);
        if (boost::filesystem::is_directory(p))
            return boost::filesystem::remove_all(p);
        return boost::filesystem::remove(p);
    }

    // @brief copy file
    static bool CopyFile(const std::string &from, const std::string &to) {
        boost::filesystem::path src(from);
        boost::filesystem::path dst(to);
        boost::filesystem::copy_file(
            src, dst, boost::filesystem::copy_option::overwrite_if_exists);
        return true;
    }

    // @brief get files in folder
    static bool GetFilesInFolder(const std::string &folder,
                                 std::vector<std::string> *ret) {
        if (!IsExists(folder) || !IsDirectory(folder)) {
		std::cout << "path is not exist or directory.";
            return false;
        }
        boost::filesystem::directory_iterator it(folder);
        boost::filesystem::directory_iterator endit;
        while (it != endit) {
            if (IsRegularFile(it->path().string())) {
                ret->push_back(it->path().string());
            }
            ++it;
        }
        std::sort(ret->begin(), ret->end());
        return true;
    }

    // @brief get files in folder recursive
    static bool GetFilesInFolderRecursive(const std::string &folder,
                                          std::vector<std::string> *ret) {
        std::cout << "folder: " << folder << std::endl;
        if (!IsExists(folder) || !IsDirectory(folder)) {
		std::cout << "path is not exist or directory.";
            return false;
        }
        boost::filesystem::recursive_directory_iterator it(folder);
        boost::filesystem::recursive_directory_iterator endit;
        while (it != endit) {
            if (IsRegularFile(it->path().string())) {
                ret->insert(ret->end(), it->path().string());
            }
            ++it;
        }

        std::sort(ret->begin(), ret->end());
        std::cout << "Load " << ret->size() << " files." << std::endl;

        return true;
    }

    // @brief get folders in folder
    static bool GetFoldersInFolder(const std::string &folder,
                                   std::vector<std::string> *ret) {
        if (!IsExists(folder) || !IsDirectory(folder)) {
		std::cout << "path is not exist or directory.";
            return false;
        }
        boost::filesystem::directory_iterator it(folder);
        boost::filesystem::directory_iterator endit;
        while (it != endit) {
            if (IsDirectory(it->path().string())) {
                ret->push_back(it->path().string());
            }
            ++it;
        }
        std::sort(ret->begin(), ret->end());
        return true;
    }

    // @brief get folders in folder recursive
    static bool GetFoldersInFolderRecursive(const std::string &folder,
                                            std::vector<std::string> *ret) {
        if (!IsExists(folder) || !IsDirectory(folder)) {
		std::cout << "path is not exist or directory.";
            return false;
        }
        boost::filesystem::recursive_directory_iterator it(folder);
        boost::filesystem::recursive_directory_iterator endit;
        while (it != endit) {
            if (IsDirectory(it->path().string())) {
                ret->push_back(it->path().string());
            }
            ++it;
        }
        std::sort(ret->begin(), ret->end());
        return true;
    }

    // @brief delete files in specific folder recursive
    static bool DeleteFilesInFolderRecursive(const std::string &folder) {
        if (!IsExists(folder) || !IsDirectory(folder)) {
		std::cout << "path is not exist or directory.";
            return false;
        }
        std::vector<std::string> files_in_folder;
        FileUtil::GetFilesInFolderRecursive(folder, &files_in_folder);
        for (const auto &filename : files_in_folder) {
            if (!DeleteFile(filename)) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace FileUtils
