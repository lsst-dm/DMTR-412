#!/usr/bin/env python3

"""
This script will summarize the DM verification completeness
based on a csv dump from the LVV project
"""
from typing import Any

from math import pi, cos, sin
import numpy as np
import pandas as pd
import warnings

from tabulate import tabulate
from prettytable import PrettyTable

from bokeh.palettes import Category20c, Viridis256, Category10
from bokeh.transform import cumsum
from bokeh.plotting import figure, output_file, save

from bokeh.io import show
from bokeh.models import (AnnularWedge, ColumnDataSource,
                          Legend, LegendItem, Plot, Range1d, LabelSet)



pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

def all_strings_equal_any(lst, values):
    return np.all(np.isin(lst, values))

def any_strings_equal(lst, values):
    return np.any(np.isin(lst, values))

def any_not_all_strings_equal(lst, values):
    matches = [s for s in lst if s in values]
    return 0 < len(matches) < len(lst)

def no_strings_equals(lst, values):
    return all(s not in values for s in lst)

def v_status(lstt):
    fully_verified = ['Verified', 'Monitoring']
    verification_started = ['In Verification', 'Covered']
    not_started = ['Not Covered']
    descoped = ['Descoped']

    lst = np.concatenate(lstt.values).tolist()
    if all_strings_equal_any(lst, fully_verified):   # Requires all VEs are fully verified
        return 'Fully Verified'
    if any_not_all_strings_equal(lst, fully_verified):   # Requires at least 1 VE is fully verified
        return 'Partially Verified'
    if any_strings_equal(lst, verification_started) and no_strings_equals(lst, fully_verified):
        return 'In Verification'
    if all_strings_equal_any(lst, not_started):
        return 'Not Started'
    if all_strings_equal_any(lst, descoped):
        return 'Descoped'
    return False

def verification_status(ves):
    """
    Checks all the VEs for a requirement
    descoped, monitoring, verified
    :return:
    """

    # Aggregate VE status to a single Req status
    req_status = ves.groupby('ReqId').agg(
        req_status = ('Status', v_status)
    ).reset_index()
   # print(req_status)

    return req_status

def plot_donut(df, title):
    xdr = Range1d(start=-2, end=2)
    ydr = Range1d(start=-2, end=2)

    plot = Plot(x_range=xdr, y_range=ydr)
    plot.title.text = "Web browser market share (November 2013)"
    plot.toolbar_location = None

    colors = {
        "Chrome": "seagreen",
        "Firefox": "tomato",
        "Safari": "orchid",
        "Opera": "firebrick",
        "IE": "skyblue",
        "Other": "lightgray",
    }

    # aggregated = df.groupby("status").sum(numeric_only=True)
    # selected = aggregated[aggregated.Share >= 1].copy()
    # selected.loc["Other"] = aggregated[aggregated.Share < 1].sum()
    browsers = df.index.tolist()

    angles = df.Share.map(lambda x: 2 * pi * (x / 100)).cumsum().tolist()

    browsers_source = ColumnDataSource(dict(
        start=[0] + angles[:-1],
        end=angles,
        colors=[colors[browser] for browser in browsers],
    ))

    glyph = AnnularWedge(x=0, y=0, inner_radius=0.9, outer_radius=1.8,
                         start_angle="start", end_angle="end",
                         line_color="white", line_width=3, fill_color="colors")
    r = plot.add_glyph(browsers_source, glyph)

    legend = Legend(location="center")
    for i, name in enumerate(colors):
        legend.items.append(LegendItem(label=name, renderers=[r], index=i))
    plot.add_layout(legend, "center")

    # Specify the output file
    file_name = f"plot_donut_{title}.html"
    output_file(file_name)
    save(plot)

def plot(df, title = "Plot"):

    df['angle'] = df['count']/df['count'].sum() * 2*pi
    df['color'] = Category10[len(df)]

    # Remove zero lines
    df_filtered = df[df['count'] != 0]

    total_reqs = df_filtered['count'].sum()
    df_filtered['anno'] = (
        df_filtered.agg(lambda row: f'{row["status"]}: {row["count"]}/{total_reqs} ({row["percentage"]}%)',
                        axis=1))

    # Calculate the position of the labels
    df_filtered['label_angle'] = df_filtered['angle'].cumsum() - df_filtered['angle'] / 2

    # Create ColumnDataSource
    source = ColumnDataSource(df_filtered)

    # Create the plot
    p = figure(height=400, title=f"DM Requirement Verification Status: {title}", tools="hover",
               toolbar_location=None, tooltips="@status: @count",
               x_range=(-0.5, 1),
               y_range=(0, 2))

    # Draw the wedges
    p.wedge(x=0, y=1, radius=0.4,
            start_angle=cumsum('angle', include_zero=True),
            end_angle=cumsum('angle'),
            line_color="white", fill_color='color',
            legend_field='anno',
            source=source)

    # Add labels inside the pie chart
    labels = LabelSet(x=0, y=1, text='anno', angle='label_angle',
                      source=source,
                      # render_mode='canvas',
                      text_align='center',
                      text_font_size="10pt", text_color="white")

    p.add_layout(labels)

    p.axis.axis_label = None
    p.axis.visible = False
    p.grid.grid_line_color = None

    # Specify the output file
    file_name = f"plot_{title}.html"
    output_file(f"../output/{file_name}")
    save(p)


def export_table(df,  title):

    table = PrettyTable()
    table.field_names = ['ReqId', 'percent_complete']

    for index, row in df.iterrows():
        table.add_row([row[col] for col in table.field_names])

    # Print the table to a text file
    with open(f'reqs_percentage_{title}.txt', 'w') as f:
        f.write(str(table))


def main(df: object, priority: object = None, title: object = None) -> object:
    """
    Summarize the verification completeness
    :param data:
    :param priority: None mean process all
    :return:
    """

    # Filter data to process according to priority specified
    if priority is None:
        p1a = df
    else:
        p1a = df[df['Priority'] == priority]


    print(p1a.head())
    print(f"Number of P{priority} LVV = {len(p1a)}")

    # Number of unique requirements
    uniq_req_ids = p1a['ReqId'].unique()
    print(f"Number of P{priority} requirements = {len(uniq_req_ids)}")

    # Sum up the VEs to compute the verification status
    sum_ve_per_req = p1a.groupby('ReqId')['VE'].count().reset_index()
    #print(sum_ve_per_req)

    ves_per_req = p1a.groupby('ReqId')['Requirement'].apply(list).reset_index()
    #print(ves_per_req)

    ves_per_req = p1a.groupby('ReqId').agg(
        reqs = ('Requirement', list),
        status = ('Status', list)
    )
    ves_per_req.reset_index(inplace=True)
    ves_per_req.columns = ['ReqId', 'VEs', 'Status']

    # Add with verification percentage of each requirement
    ves_per_req['percent_complete'] =  ves_per_req['Status'].apply(
        lambda x: round(sum(1 for item in x if item in ['Verified', 'Monitoring']) / len(x) * 100),2)
    # print(tabulate(ves_per_req, headers='keys', tablefmt='psql'))



    # Hack until the MD model is updated
    ves_per_req.loc[ves_per_req['ReqId'] == 'DMS-REQ-0298', 'percent_complete'] = 67
    ves_per_req.loc[ves_per_req['ReqId'] == 'DMS-REQ-0158', 'percent_complete'] = 67
    export_table(ves_per_req,  title = f"P{priority}")


    # Compute the verification status per requirement and add to summary table
    verif_status = verification_status(ves_per_req)

    # Summarize
    aggregated = verif_status.groupby('req_status').count().reset_index()
    aggregated.rename(columns={'req_status': 'status', 'ReqId': 'count'}, inplace=True)

    # Add in missing status so that colors in plots are consistent
    status_order = ['Fully Verified', 'Partially Verified', 'In Verification', 'Verification Started', 'Not Started', 'Descoped']
    if len(aggregated) < len(status_order):
        missing_values = [value for value in status_order if value not in aggregated['status'].values]
        for mv in missing_values:
            new_row = pd.DataFrame({'status': mv, 'count': [0]})
            aggregated = pd.concat([aggregated, new_row], ignore_index=True)

    # Sort columns alphabetically using sort_index
    aggregated['status'] = pd.Categorical(aggregated['status'], categories=status_order, ordered=True)
    aggregated = aggregated.sort_values('status')

    print(f"Priority {priority} Summary: ")

    # Add a percentage of total column
    total_reqs = aggregated['count'].sum()
    aggregated['percentage'] = ((aggregated['count'] / total_reqs) * 100).round(1)
    print(aggregated)

    # Plot pie chart
    plot(aggregated, f"{title}_P{priority}")
    #plot_donut(aggregated, f"P{priority}")

def read_csv(csv_file):
    """
    Read in the csv dump from LVV
    :param data_path:
    :return:
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file)

    # Change some column names
    df = df.rename(columns={'Custom field (Requirement Priority)': 'Priority'})

    # Add some manual updates not yet in the model - DMSR only
    # df.loc[df['Issue key'] == 'LVV-138', 'Priority'] = ['2']

    # Split out the Requirement Id from the LVV element
    df[['Requirement', 'Description']] = df['Summary'].str.split(':\s*', n=1, expand=True)
    df[['ReqId', 'VE']] = df['Requirement'].str.split('-V-', expand=True)

    # Add some checks later
    return df


def summarize_priority(df):
    """
    Group all unique requirements by priority and count
    :param data:
    :return:
    """
    gp = df.groupby('Priority')['ReqId'].nunique().reset_index()
    print(f"Number of unique requirements per prioritization")
    print(gp)
    pass


if __name__ == '__main__':

    # DMSR
    data_path ="../data/dmsr-lvv-20241029.csv"
    data = read_csv(data_path)

    summarize_priority(data)
    main(data, '1a', title = "dm")
    main(data, '1b', title = "dm")
    main(data, '2', title = "dm")
    main(data, '3', title = "dm")

    # Middleware
    print(f"Middleware")
    mw_data_path ="../data/dmmw-lvv-20241028.csv"
    mw_data = read_csv(mw_data_path)
    main(mw_data, '1a', title = "mw")
    main(mw_data, '1b', title = "mw")
    main(mw_data, '2', title = "mw")

    # RSP
    print(f"Science Platform")
    rsp_data_path = "../data/dmlsp-lvv-20241029.csv"
    rsp_data = read_csv(rsp_data_path)
    main(rsp_data, title='rsp')