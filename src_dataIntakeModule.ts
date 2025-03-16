/**
 * Data Intake Module
 * 
 * Collects structured data from various internal business systems such as 
 * ERP, CRM, IoT sensors, and financial records.
 */

import axios from 'axios';
import fs from 'fs';
import path from 'path';
import winston from 'winston';
import dotenv from 'dotenv';

dotenv.config();

// Configure logger
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json()
  ),
  defaultMeta: { service: 'data-intake-module' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      ),
    }),
  ],
});

// Supported data source types
export enum DataSourceType {
  ERP = 'erp',
  CRM = 'crm',
  IOT = 'iot',
  FINANCIAL = 'financial',
  CUSTOM = 'custom',
}

// Interface for data source configuration
export interface DataSourceConfig {
  type: DataSourceType;
  name: string;
  endpoint: string;
  method: 'GET' | 'POST';
  headers?: Record<string, string>;
  body?: any;
  parameters?: Record<string, string>;
  authType?: 'none' | 'basic' | 'bearer' | 'apiKey';
  authConfig?: {
    username?: string;
    password?: string;
    token?: string;
    apiKeyName?: string;
    apiKeyValue?: string;
    apiKeyLocation?: 'header' | 'query';
  };
  transformationScript?: string;
  schedule?: string; // cron expression
  enabled: boolean;
}

export class DataIntakeModule {
  private dataSourceConfigs: DataSourceConfig[] = [];
  private dataDirectory: string;
  
  constructor(configPath?: string) {
    // Create data directories if they don't exist
    this.dataDirectory = path.join(process.cwd(), 'data');
    if (!fs.existsSync(this.dataDirectory)) {
      fs.mkdirSync(this.dataDirectory, { recursive: true });
    }
    
    const rawDataDir = path.join(this.dataDirectory, 'raw');
    if (!fs.existsSync(rawDataDir)) {
      fs.mkdirSync(rawDataDir, { recursive: true });
    }
    
    const processedDataDir = path.join(this.dataDirectory, 'processed');
    if (!fs.existsSync(processedDataDir)) {
      fs.mkdirSync(processedDataDir, { recursive: true });
    }
    
    // Load configuration if provided
    if (configPath && fs.existsSync(configPath)) {
      try {
        const configFile = fs.readFileSync(configPath, 'utf8');
        this.dataSourceConfigs = JSON.parse(configFile);
        logger.info(`Loaded ${this.dataSourceConfigs.length} data source configurations`);
      } catch (error) {
        logger.error('Error loading configuration file', { error });
        throw new Error('Failed to load configuration file');
      }
    }
  }
  
  /**
   * Add a new data source configuration
   * @param config The data source configuration to add
   */
  public addDataSource(config: DataSourceConfig): void {
    this.dataSourceConfigs.push(config);
    logger.info(`Added new data source: ${config.name} (${config.type})`);
  }
  
  /**
   * Remove a data source by name
   * @param name The name of the data source to remove
   */
  public removeDataSource(name: string): boolean {
    const initialLength = this.dataSourceConfigs.length;
    this.dataSourceConfigs = this.dataSourceConfigs.filter(config => config.name !== name);
    
    if (initialLength !== this.dataSourceConfigs.length) {
      logger.info(`Removed data source: ${name}`);
      return true;
    }
    
    logger.warn(`Data source not found: ${name}`);
    return false;
  }
  
  /**
   * Update an existing data source configuration
   * @param name The name of the data source to update
   * @param updatedConfig The updated configuration
   */
  public updateDataSource(name: string, updatedConfig: Partial<DataSourceConfig>): boolean {
    const index = this.dataSourceConfigs.findIndex(config => config.name === name);
    
    if (index === -1) {
      logger.warn(`Data source not found: ${name}`);
      return false;
    }
    
    this.dataSourceConfigs[index] = { ...this.dataSourceConfigs[index], ...updatedConfig };
    logger.info(`Updated data source: ${name}`);
    return true;
  }
  
  /**
   * Save the current data source configurations to a file
   * @param filePath The path to save the configuration file
   */
  public saveConfiguration(filePath: string): void {
    try {
      fs.writeFileSync(filePath, JSON.stringify(this.dataSourceConfigs, null, 2));
      logger.info(`Saved configuration to ${filePath}`);
    } catch (error) {
      logger.error('Error saving configuration file', { error });
      throw new Error('Failed to save configuration file');
    }
  }
  
  /**
   * Collect data from all enabled data sources
   */
  public async collectAllData(): Promise<void> {
    logger.info('Starting data collection from all enabled sources');
    
    const enabledSources = this.dataSourceConfigs.filter(config => config.enabled);
    
    if (enabledSources.length === 0) {
      logger.warn('No enabled data sources found');
      return;
    }
    
    for (const config of enabledSources) {
      try {
        await this.collectDataFromSource(config);
      } catch (error) {
        logger.error(`Error collecting data from ${config.name}`, { error });
      }
    }
    
    logger.info('Completed data collection from all enabled sources');
  }
  
  /**
   * Collect data from a specific source
   * @param config The data source configuration
   */
  public async collectDataFromSource(config: DataSourceConfig): Promise<any> {
    logger.info(`Collecting data from ${config.name} (${config.type})`);
    
    try {
      // Prepare request configuration
      const requestConfig: any = {
        method: config.method,
        url: config.endpoint,
        headers: { ...config.headers },
      };
      
      // Handle authentication
      if (config.authType !== 'none' && config.authConfig) {
        switch (config.authType) {
          case 'basic':
            if (config.authConfig.username && config.authConfig.password) {
              requestConfig.auth = {
                username: config.authConfig.username,
                password: config.authConfig.password,
              };
            }
            break;
          case 'bearer':
            if (config.authConfig.token) {
              requestConfig.headers.Authorization = `Bearer ${config.authConfig.token}`;
            }
            break;
          case 'apiKey':
            if (config.authConfig.apiKeyName && config.authConfig.apiKeyValue) {
              if (config.authConfig.apiKeyLocation === 'header') {
                requestConfig.headers[config.authConfig.apiKeyName] = config.authConfig.apiKeyValue;
              } else if (config.authConfig.apiKeyLocation === 'query') {
                requestConfig.params = {
                  ...requestConfig.params,
                  [config.authConfig.apiKeyName]: config.authConfig.apiKeyValue
                };
              }
            }
            break;
        }
      }
      
      // Add query parameters if any
      if (config.parameters) {
        requestConfig.params = { ...config.parameters };
      }
      
      // Add request body if needed
      if (config.method === 'POST' && config.body) {
        requestConfig.data = config.body;
      }
      
      // Make the request
      const response = await axios(requestConfig);
      
      // Process and save the data
      let data = response.data;
      
      // Apply transformation if specified
      if (config.transformationScript) {
        try {
          // Load and execute the transformation script
          const transform = require(path.resolve(config.transformationScript));
          data = transform(data);
        } catch (transformError) {
          logger.error(`Error applying transformation for ${config.name}`, { error: transformError });
        }
      }
      
      // Save raw data
      const timestamp = new Date().toISOString().replace(/:/g, '-');
      const rawFilePath = path.join(this.dataDirectory, 'raw', `${config.name}_${timestamp}.json`);
      
      fs.writeFileSync(rawFilePath, JSON.stringify(data, null, 2));
      logger.info(`Saved raw data from ${config.name} to ${rawFilePath}`);
      
      return data;
    } catch (error) {
      logger.error(`Failed to collect data from ${config.name}`, { error });
      throw error;
    }
  }
  
  /**
   * List all configured data sources
   */
  public listDataSources(): DataSourceConfig[] {
    return this.dataSourceConfigs;
  }
}

// Export singleton instance
export default new DataIntakeModule();