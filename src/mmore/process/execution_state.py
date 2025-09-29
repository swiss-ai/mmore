import logging
from typing import Optional, cast

from dask.distributed import Variable

logger = logging.getLogger(__name__)


class ExecutionState:
    """
    The global state of the execution, this class is static and stores the global state whenever the execution should stop or not.
    Every processor can check this state to see if it should stop execution.
    Supports both local and distributed execution
    In local execution, the state is stored in a static variable
    In distributed execution, the state is stored in a dask 'Variable',
    you need to connect to a dask cluster to use this feature (use client = Client(...) before calling .initialize())
    and this variable will be shared across the custer
    """

    # static variables
    _use_dask: Optional[bool] = None
    _dask_var: Optional[Variable] = None
    _local_state: bool = False

    @staticmethod
    def initialize(distributed_mode=False, client=None):
        """
        Initializes the global state manager
        :param distributed_mode: Whether the execution is in distributed mode
        :param client: connection client to the dask cluster
        """
        if ExecutionState._use_dask is not None:
            raise Exception("Execution state already initialized")
        assert distributed_mode is not None, (
            "Distributed mode must be set to True or False"
        )
        ExecutionState._use_dask = distributed_mode

        if distributed_mode:
            assert client is not None, (
                "You must be in the context of a dask client to use distributed mode"
            )
            ExecutionState._dask_var = Variable("should_stop_execution", client=client)
            ExecutionState._dask_var.set(False)
            logger.info("Execution state initialized (distributed mode)")
        else:
            ExecutionState._local_state = False
            logger.info("Execution state initialized (local mode)")

    @staticmethod
    def shutdown():
        ExecutionState._use_dask = None
        ExecutionState._dask_var = None
        ExecutionState._local_state = False

    @staticmethod
    def get_should_stop_execution() -> bool:
        """Returns the global execution state (True if it should stop)"""
        if ExecutionState._use_dask is None:
            raise Exception("Execution state not initialized")
        if ExecutionState._use_dask:
            try:
                return cast(bool, cast(Variable, ExecutionState._dask_var).get())
            except Exception as e:
                logger.error(f"Error getting dask variable: {e}")
                return True
        else:
            return ExecutionState._local_state

    @staticmethod
    def set_should_stop_execution(value: bool):
        """Sets the global execution stop state"""
        logger.info(f"Setting execution state to {value}")
        if ExecutionState._use_dask is None:
            raise Exception("Execution state not initialized")
        if ExecutionState._use_dask:
            cast(Variable, ExecutionState._dask_var).set(value)
        else:
            ExecutionState._local_state = value
        logger.info(f"Execution state set to {value}")
