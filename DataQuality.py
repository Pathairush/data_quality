import datetime
import pytz

import os
import json
from ruamel import yaml
import ruamel
from collections import defaultdict
import re

import pandas as pd
import pyspark.sql.functions as F
import databricks.koalas as ks
import numpy as np

import great_expectations 
from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.checkpoint.checkpoint import SimpleCheckpoint
from great_expectations.data_context import BaseDataContext
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)
from great_expectations.profile.user_configurable_profiler  import UserConfigurableProfiler

class DataQuality():
    
    def __init__(self, datasource_name, spark_dataframe, partition_date: datetime.date):
        
        '''
        create great expectations context and default runtime datasource
        '''
        
        self.datasource_name = datasource_name
        self.expectation_suite_name = f"{datasource_name}_expectation_suite"
        self.checkpoint_name = f"{datasource_name}_checkpoint"
        self.spark_dataframe = spark_dataframe
        self.partition_date = partition_date
        
        # create data context
        root_directory = "/dbfs/great_expectations/"
        data_context_config = DataContextConfig( store_backend_defaults=FilesystemStoreBackendDefaults(root_directory=root_directory) )
        context = BaseDataContext(project_config = data_context_config)
                   
        datasource_yaml = rf"""
        name: {self.datasource_name}
        class_name: Datasource
        execution_engine:
            class_name: SparkDFExecutionEngine
        data_connectors:
            runtime_connector:
                class_name: RuntimeDataConnector
                batch_identifiers:
                    - run_id
        """
        context.test_yaml_config(datasource_yaml)
        context.add_datasource(**yaml.load(datasource_yaml, Loader=ruamel.yaml.Loader))
        
        self.context = context
    
    def get_context(self):
        
        '''
        retrieving data context in case you would like to manually extract / tweak the context by yourself
        '''
        
        return self.context
    
    def get_expectation_suit(self):
        
        '''
        retriving the current expectation suite
        '''
        
        return self.context.get_expectation_suite(self.expectation_suite_name)
    
    def create_batch_data(self, df, partition_date: datetime.date):
        
        '''
        create runtime batch request from the input spark dataframe and partition date
        '''
        
        batch_request = RuntimeBatchRequest(
            datasource_name= self.datasource_name,
            data_connector_name= "runtime_connector",
            data_asset_name=f"{self.datasource_name}_{self.partition_date.strftime('%Y%m%d')}",
            batch_identifiers={
                "run_id": f'''
                {self.datasource_name}_partition_date={self.partition_date.strftime('%Y%m%d')}_runtime={datetime.datetime.today().strftime('%Y%m%d')}_user={self._get_current_user()}
                ''',
            },
            runtime_parameters={"batch_data": df}
        )
        
        return batch_request
    
    def create_expectation_suite_if_not_exist(self):
        
        '''
        create expectation suite if not exist
        '''
        
        try:
            # create expectation suite
            self.context.create_expectation_suite(
                expectation_suite_name = self.expectation_suite_name,
                overwrite_existing=False
            )
        except great_expectations.exceptions.DataContextError as e:
            print(e)
        except Error as e:
            raise e
            
    def delete_expectation_suite(self):
        
        '''
        delete the expectation suite
        '''
        
        self.context.delete_expectation_suite(expectation_suite_name = self.expectation_suite_name)
    
    def get_validator(self, with_profiled=False):
        
        '''
        retreiving a validator object for a fine grain adjustment on the expectation suite.
        '''
        
        batch_request = self.create_batch_data(self.spark_dataframe, self.partition_date)
        self.create_expectation_suite_if_not_exist()

        validator = self.context.get_validator(
            batch_request = batch_request,
            expectation_suite_name = self.expectation_suite_name,
        )
        
        if with_profiled:

            # build expectation with profiler
            not_null_only = True
            table_expectations_only = False

            profiler = UserConfigurableProfiler(
                profile_dataset = validator,
                not_null_only = not_null_only,
                table_expectations_only = table_expectations_only
            )

            suite = profiler.build_suite()

            # save validation
            validator.save_expectation_suite(discard_failed_expectations=False)
        
        return validator
    
    def create_checkpoint_if_not_exist(self):
        
        '''
        create checkpoint if not exist.
        '''
        
        try:
            self.context.get_checkpoint(self.checkpoint_name)
            print(f'{self.checkpoint_name} is already created')
                    
        except great_expectations.exceptions.CheckpointNotFoundError:
            
            checkpoint_config = {
                "name": self.checkpoint_name,
                "config_version": 1.0,
                "class_name": "SimpleCheckpoint",
                "run_name_template": "%Y%m%d-%H%M%S",
            }
            self.context.test_yaml_config(yaml.dump(checkpoint_config))
            self.context.add_checkpoint(**checkpoint_config)

        except Error as e:
            raise e
            
            
    def validate_data(self, df=None, partition_date: datetime.date=None):
        
        '''
        validate dataset using the input dataset when initiated the class
        or user provided dataset when calling the method.
        '''
        
        if df and partition_date:
            batch_request = self.create_batch_data(df, partition_date)
        else:
            batch_request = self.create_batch_data(self.spark_dataframe, self.partition_date)            
        
        self.create_checkpoint_if_not_exist()
        
        # run expectation_suite against data
        checkpoint_result = self.context.run_checkpoint(
            checkpoint_name = self.checkpoint_name,
            validations=[
                {
                    "batch_request": batch_request,
                    "expectation_suite_name": self.expectation_suite_name,
                }
            ],
        )
        
        for k,v in checkpoint_result['run_results'].items():
            self.render_file = v['actions_results']['update_data_docs']['local_site'].replace('file://', '')
    
        return checkpoint_result
    
    def render_report(self, to_render_file=None):
        
        '''
        render report from the validation result
        required user to trigger `validate_data` at least once.
        or render report from the `to_render_file` absolute path.
        '''
        
        try:
            if not to_render_file:
                to_render_file = self.render_file
        except AttributeError:
            raise ValueError("The render file doesn't exists, please call the `validate_data` method to get a render file")
        except Error as e:
            raise e
                
        with open(to_render_file, "r", encoding='utf-8') as f:
            text= f.read()

        displayHTML(text)
        
    def render_expectation_report(self):
        
        '''
        retreiving the current expectation suite.
        '''
        
        expectation_path = f'/dbfs/great_expectations/uncommitted/data_docs/local_site/expectations/{self.expectation_suite_name.replace(".", "/")}.html'
        
        with open(expectation_path, "r", encoding='utf-8') as f:
            text= f.read()

        displayHTML(text)
        
    def backup_great_expectations_db(self, persisted_target_dir):
        
        '''
        back up great expectation database to azure blob storage.
        '''
        
        dbutils.fs.cp('dbfs:/great_expectations', persisted_target_dir, recurse=True)
        
    def _get_current_user(self):
        
        '''
        retreiving the current username who run the code.
        '''
        
        response = dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson() 
        response = json.loads(response)
        return  response['tags']['user']
    
    def _check_sdf_completeness(self):
        
        try:
            self.spark_dataframe.count()
            return True
        except Exception as e:
            print("the spark dataframe `count` method returns error, please check the dataframe completeness.")
            raise e
      
    def _extract_lineage(self):
        
        return self.spark_dataframe._sc._jvm.PythonSQLUtils.explainString(self.spark_dataframe.queryExecution(), "formatted")

    def get_data_lineage(self):
        
        if self._check_sdf_completeness():
            
            lineage_txt = self._extract_lineage()

            pttrn_scan_tbl = r"\([0-9]*\)\sScan\s([a-z]*)\s([a-z._0-9]*)(\n.*){,6}"
            pttrn_scan_tbl_prop = r"(?:.*)\:(?:.*){,6}"
            pttrn_kv = r"(\w*).*\:[\s](.*)"

            result = defaultdict(list)
            scan_tables = re.finditer(pttrn_scan_tbl, lineage_txt)

            if scan_tables:

                for table in scan_tables:

                    result['source_table_file_type'].append(table.group(1))
                    result['source_table_name'].append(table.group(2))

                    # extract properties
                    match_table_string = table[0]
                    props = re.findall(pttrn_scan_tbl_prop, match_table_string)

                    expect_props = ['Output', 'Batched', 'Location', 'PartitionFilters', 'ReadSchema', 'PushedFilters']
                    avaiable_props = [re.sub(r'[^A-Za-z]', '', e.split(':')[0])  for e in props]

                    # impute the unavailable props
                    for exp in expect_props:
                        if exp not in avaiable_props:
                            props.append(f"{exp}: []")

                    for prop in props:
                        kv = re.match(pttrn_kv, prop)
                        result[kv.group(1)].append(kv.group(2))

            self.data_lineage_dict = result

            return result
    
    def convert_data_lineage_dict_to_sdf(self):
        
        result_df = ks.DataFrame(self.data_lineage_dict)
        result_df = result_df.to_spark()

        # clean data 
        result_df = result_df.withColumn('source_input_columns', F.split(F.regexp_replace(F.col('Output'), r'(?:\#[0-9]+)|(?:\[)|(?:\])|(?:\s)', ''), ',')).drop('Output')
        result_df = result_df.withColumn('source_num_input_columns', F.size('source_input_columns'))

        # format column name
        result_df = result_df.toDF(*[c if c.startswith('source') else 'source_' + c.lower() for c in result_df.columns ])

        # add information
        result_df = result_df.withColumn('output_table', F.lit(self.datasource_name))
        result_df = result_df.withColumn('update_dt', F.lit(str(datetime.datetime.now(pytz.timezone('Asia/Bangkok')))))
        
        self.data_lineage_sdf = result_df
        
        return result_df
    
    def update_data_lineage(self, persisted_target_dir):
        
        self.get_data_lineage()
        self.convert_data_lineage_dict_to_sdf()
        
        if self.data_lineage_sdf:
        
            (
                self.data_lineage_sdf
                .write
                .format('delta')
                .mode('append')
                .save(persisted_target_dir)
            )
            
            print(f'the data lineage for {self.datasource_name} has been updated')