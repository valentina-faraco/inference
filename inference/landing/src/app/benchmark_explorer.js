'use client'

import { Box, Grid, Typography } from "@mui/material";
import { DataGrid } from '@mui/x-data-grid';
import React from "react";

const columns = [
    { field: 'confidence', type: 'number', headerName: 'Confidence Threshold', width: 50 },
    {
        field: 'image_size',
        headerName: 'Original Image Size',
        width: 150,
        editable: true,
    },
    {
        field: 'interface',
        headerName: 'Interface',
        width: 100,
        editable: true,
    },
    {
        field: 'iou_threshold',
        headerName: 'IoU Threshold',
        type: 'number',
        width: 50,
        editable: true,
    },
    {
        field: 'num_iterations',
        headerName: 'Number of Iterations',
        type: 'number',
        width: 100,
        editable: true,
    },
    {
        field: 'num_warmup_iterations',
        headerName: 'Number of Warmup Iterations',
        type: 'number',
        width: 100,
        editable: true,
    },
    {
        field: 'model_input_size',
        headerName: 'Model Input Size',
        width: 150,
        editable: true,
    },
    {
        field: 'task',
        headerName: 'Task',
        width: 150,
        editable: true,
    },
    {
        field: 'rps',
        headerName: 'RPS',
        type: 'number',
        width: 100,
        editable: true,
    },
    {
        field: 'fps',
        headerName: 'FPS',
        type: 'number',
        width: 100,
        editable: true,
    },
];

export const BenchmarkExplorer = () => {
    const [rows, setRows] = React.useState([]);
    React.useEffect(() => {
        fetch('http://localhost:9001/benchmark_data')
            .then((response) => response.json())
            .then((data) => {
                setRows(data.map((rows, index) => {
                    return { ...rows, id: index }
                }))
            });
    }, [])
    return (
        <Box width={'100%'}>
            <Grid container>
                <Grid item xs={12}>
                    <Typography variant="h5">Benchmark Explorer</Typography>
                </Grid>
                <Grid item xs={12}>
                    <DataGrid
                        disableVirtualization
                        // width="100%"
                        rows={rows}
                        columns={columns}
                        initialState={{
                            pagination: {
                                paginationModel: {
                                    pageSize: 50,
                                },
                            },
                        }}
                        pageSizeOptions={[5]}
                        checkboxSelection
                        disableRowSelectionOnClick
                    />
                </Grid>
            </Grid>
        </Box>
    );
}

