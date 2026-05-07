import multiprocessing

class MultiProcessRuntime():

    def __init__(self, runtime, config, process_num=4, simulation=True):
        self.runtime = runtime
        self.config = config
        self.process_num = process_num
        self.simulation = simulation

    def calculate(self, ir, inputs, kwargs):
        rt = self.runtime(**self.config)
        rt.init_rpc(ir, simulation=self.simulation)
        result = getattr(rt, 'run_ir')(ir, inputs, **kwargs)
        return result

    def run_data_parallel(self, mapped_ir, in_value, *, weights= None,
                          outputs=None, callback=None, logfile = None,
                          ):
        # Validate IR input count matches provided inputs.
        assert len(mapped_ir) == len(in_value)

        PROCESSES = self.process_num
        print('Creating pool with %d processes\n' % PROCESSES)
        kwargs = {}
        if weights != None:
            kwargs.update({"weights":weights})
        if outputs != None:
            kwargs.update({"outputs":outputs})
        if callback != None:
            kwargs.update({"callback":callback})
        if logfile != None:
            kwargs.update({"logfile":logfile})

        re_value = []
        with multiprocessing.Pool(PROCESSES) as pool:
            # Create worker tasks.
            TASKS = []
            index = 0
            for i in in_value:
                TASKS.append((mapped_ir[index], i, kwargs))
                index += 1
            assert TASKS != []
        # Collect results in order.
            results = [pool.apply_async(self.calculate, t) for t in TASKS]
            for r in results:
                re_value.append(r.get())
        return re_value

    def pipe_calculate(self, ir, input_q, output_q, kwargs, final):
        rt = self.runtime(**self.config)
        rt.init_rpc(ir, simulation=self.simulation)
        while True:
            inputs = input_q.get()
            # Process the data for each stage
            if inputs is None:
                if not final: # final re did not need to put None in queue
                    output_q.put(inputs)
                break
            result = getattr(rt, 'run_ir')(ir, inputs, **kwargs)
            if final:
                output_q.append(result)
            else:
                output_q.put(result)

    def run_model_parallel(self, mapped_ir, in_value, *, weights= None,
                          outputs=None, callback=None, logfile = None,):

        # Validate output count matches process count.
        assert len(outputs) == self.process_num

        # kwargs
        kwargs_ = []
        assert len(weights) == len(outputs)
        assert len(outputs) == len(callback)
        assert len(callback) == len(logfile)
        for i in range(len(weights)):
            kwargs = {}
            if weights[i] != None:
                kwargs.update({"weights":weights[i]})
            if outputs[i] != None:
                kwargs.update({"outputs":outputs[i]})
            if callback[i] != None:
                kwargs.update({"callback":callback[i]})
            if logfile[i] != None:
                kwargs.update({"logfile":logfile[i]})
            kwargs_.append(kwargs)

        # Initialize result list.
        results = multiprocessing.Manager().list()

        # Initialize processes.
        PROCESSES = self.process_num
        print('Creating pool with %d processes\n' % PROCESSES)

        # Create queues.
        queue_list = []
        for i in range(self.process_num):
            queue_list.append(multiprocessing.Queue())

        # Spawn processes.
        process_list = []
        for j in range(self.process_num):
            if j == self.process_num - 1:
                process_list.append(multiprocessing.Process(target=self.pipe_calculate,
                                                            args=(mapped_ir, queue_list[j],
                                                                  results, kwargs_[j],
                                                                  True
                                                                  )
                                                            )
                                    )
            else:
                process_list.append(multiprocessing.Process(target=self.pipe_calculate,
                                                            args=(mapped_ir, queue_list[j],
                                                                  queue_list[j+1], kwargs_[j],
                                                                  False)
                                                            )
                                    )
        # Start execution.
        for i in range(self.process_num):
            process_list[i].start()

        # Send input data to the first queue.
        for i in in_value:
            queue_list[0].put(i)

        # Send termination signal to the first queue.
        queue_list[0].put(None)

        # Wait for completion.
        for i in range(self.process_num):
            process_list[i].join()

        # Close queues.
        for i in range(self.process_num):
            queue_list[i].close()

        return list(results)
