package com.datamate.datamanagement.infrastructure.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

import java.util.ArrayList;
import java.util.List;

/**
 * 数据管理服务配置属性
 */
@Configuration
@ConfigurationProperties(prefix = "datamate.datamanagement")
public class DataManagementProperties {

    private FileStorage fileStorage = new FileStorage();
    private Cache cache = new Cache();

    public FileStorage getFileStorage() {
        return fileStorage;
    }

    public void setFileStorage(FileStorage fileStorage) {
        this.fileStorage = fileStorage;
    }

    public Cache getCache() {
        return cache;
    }

    public void setCache(Cache cache) {
        this.cache = cache;
    }

    public static class FileStorage {
        private String uploadDir = "./uploads";
        private long maxFileSize = 10485760; // 10MB
        private long maxRequestSize = 52428800; // 50MB
        private List<String> allowedSourceDirs = new ArrayList<>();

        public String getUploadDir() {
            return uploadDir;
        }

        public void setUploadDir(String uploadDir) {
            this.uploadDir = uploadDir;
        }

        public long getMaxFileSize() {
            return maxFileSize;
        }

        public void setMaxFileSize(long maxFileSize) {
            this.maxFileSize = maxFileSize;
        }

        public long getMaxRequestSize() {
            return maxRequestSize;
        }

        public void setMaxRequestSize(long maxRequestSize) {
            this.maxRequestSize = maxRequestSize;
        }

        public List<String> getAllowedSourceDirs() {
            return allowedSourceDirs;
        }

        public void setAllowedSourceDirs(List<String> allowedSourceDirs) {
            this.allowedSourceDirs = allowedSourceDirs;
        }
    }

    public static class Cache {
        private int ttl = 3600; // 1 hour
        private int maxSize = 1000;

        public int getTtl() {
            return ttl;
        }

        public void setTtl(int ttl) {
            this.ttl = ttl;
        }

        public int getMaxSize() {
            return maxSize;
        }

        public void setMaxSize(int maxSize) {
            this.maxSize = maxSize;
        }
    }
}
