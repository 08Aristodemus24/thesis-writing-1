import { useContext, useEffect, useRef } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

import * as d3 from 'd3';

function calculateMinMaxSignal(sprSheet){
    let max_signal, min_signal, min_sec, max_sec;
            
    // if new_signal does not yet exist that means user has not
    // yet submitted and dataframe still has the same columns
    if(!('new_signal' in sprSheet[0])){
        max_signal = d3.max(sprSheet, (row) => row['raw_signal']);
        min_signal = d3.min(sprSheet, (row) => row['raw_signal']);
        min_sec = sprSheet[0]['time'];
        max_sec = sprSheet[sprSheet.length - 1]['time'];
    }

    // if the new_signal does indeed already exist then the
    // make new calculations to the max_signal and min_signal
    // as these will obviously change right after correction
    else{ 
        const max_raw_signal = d3.max(sprSheet, (row) => row['raw_signal']);
        const min_raw_signal = d3.min(sprSheet, (row) => row['raw_signal']);
        const max_clean_signal = d3.max(sprSheet, (row) => row['new_signal']);
        const min_clean_signal = d3.min(sprSheet, (row) => row['new_signal']);
        
        // get final max value between the max of the raw signal and the 
        // max of the clean signal
        max_signal = Math.max(max_raw_signal, max_clean_signal);
        min_signal = Math.max(min_raw_signal, min_clean_signal);

        min_sec = sprSheet[0]['time'];
        max_sec = sprSheet[sprSheet.length - 1]['time'];
    }

    return [max_signal, min_signal, min_sec, max_sec];
}

export default function Visualizer({ children }){

    // initialize and define theme of component by using
    // context
    let style;
    const designs = useContext(DesignsContext);
    const themes = useContext(ThemeContext);
    const { design, theme } = themes;
    
    // sometimes themes context will contain only the design 
    // and not the theme key so check if theme key is in themes
    if('theme' in themes){
        style = designs[design][theme];
    }else{
        style = designs[design];
    }

    /** where to actually display or undisplay path object when either show raw or show correct states are changed */
    let { initSprSheet, showRaw, showCorrect, showArt, showStressLevels } = useContext(FormInputsContext);

    // recall useRef() is akin in html to selecting an element via
    // the document tree object
    const svgRef = useRef();
    const svg = d3.select(svgRef.current);

    // when user uploads and when corrected df is returned
    // setInitSprSheet() updater is always called and when the
    // initSprSheet is updated this will fire
    useEffect(() => {
        // console.log("state updated");
        // console.log(initSprSheet);

        // upon upload of user of .csv file or new .csv file
        // check also if there is already an existing g child
        // under the svg element, if there is remove it so it can
        // be replaced by new g element

        /*
        initSprSheet and finalSprSheet are states that when
        component is mounted or updated this useEffect will fire

        what I want is when the initSprSheet state changes
        I want to have the option to let the user see the artifacts in
        the signal i.e. show artifact as a checkbox

        and I don't want anymore to remove the initSprSheet but rather
        update it such that it includes the new_signals column 

        maybe instead of two spreadsheets 
        */
        svg.selectAll("*").remove();
        
        // when initial spreadsheet is removed because of submit
        // and be replaced by final spreadsheet, final spreadsheet
        // will now replace this initial spreadsheet svg
        if(initSprSheet.length != 0){
            let [max_signal, min_signal, min_sec, max_sec] = calculateMinMaxSignal(initSprSheet);
            
            // calculated max and min of signals and seconds
            console.log(`max signal: ${max_signal}`);
            console.log(`min_signal: ${min_signal}`);
            console.log(`max_sec: ${max_sec}`);
            console.log(`min_sec: ${min_sec}`);

            // const width = 'clamp(500px, 75vw, 1260px)';
            // const height = '250px';
            const margin = {top: 10, right: 30, bottom: 30, left: 60, }
            const width = 800 - margin["left"] - margin["right"];
            const height = 400 - margin["top"] - margin["bottom"]; 

            // recall translate takes in x and y coordinates of how much
            // to move the element along the x and y axis respectively
            // NOTE: although the svg is selected is is not actually returned
            // as an append is used where the g element is added and so the 
            // g element is actually used here
            svg.attr("width", width + margin.left + margin.right) // still is 768 since we add back the subtracted values from margin top and margin bottom
            .attr("height", height + margin.top + margin.bottom) // still is 486 since we add back the subtracted values from margin top and margin bottom
            .attr("viewBox", [
                0,
                0,
                (width + margin["left"] + margin["right"]),
                (height + margin["top"] + margin["bottom"])]);

            const g = svg.append("g")
            .attr("class", "cartesian-plane")
            .attr("transform", `translate(${margin["left"]}, ${margin["top"]})`); // this is the g element which draws the line

            // x here is a callback function
            let x = d3.scaleLinear()
            .domain([min_sec, max_sec])
            .range([0, width]);

            let x_axis = d3.axisBottom(x)

            // we create a g element which will draw the x-axis
            let x_group = g.append('g')
            .attr("class", "x-axis")
            .attr('transform', `translate(0, ${height})`)
            
            // y here is also callback function
            let y = d3.scaleLinear()
            .domain([0, max_signal])
            .range([height, 0]);

            let y_axis = d3.axisLeft(y)

            // we create a g element which will draw the y-axis
            let y_group = g.append("g")
            .attr("class", "y-axis")

            x_group.call(x_axis)
            y_group.call(y_axis);

            // add title/label to both x and y axes
            // x axis label
            g.append("text")
            .attr("text-anchor", "middle")
            .attr("x", width / 2) // positive values makes the label go right
            .attr("y", height + margin["bottom"]) // positive values makes the label go down further
            .text("time (s)");

            // y axis label
            g.append("text")
            .attr("text-anchor", "middle")
            .attr("transform", "rotate(-90)") // rotated 90 degrees to the left in a counterclock wise manner
            .attr("y", -margin["bottom"]) // moves to the right if positive value and left if negative value
            .attr("x", -margin["left"] * 3) // moves up if positive value and down if negative value
            .text("microsiemens (μS)");

            // set the gradient for the raw signal
            g.append("linearGradient")
            .attr("class", "raw-line-gradient")
            .attr("id", "raw-line-gradient")
            .attr("gradientUnits", "userSpaceOnUse")
            .attr("x1", 0)
            .attr("y1", y(0))
            .attr("x2", 0)
            .attr("y2", y(max_signal))
            .selectAll("stop")
            .data([
                {offset: "0%", color: "#c78324"},
                {offset: "50%", color: "#ab229d"},
                {offset: "100%", color: "#2823ba"}
            ])
            .enter().append("stop")
            .attr("offset", (d) => d["offset"])
            .attr("stop-color", (d) => d["color"]);
      
            // define line generator for raw signal
            let raw_line = d3.line()
            .x(d => x(d['time']))
            .y(d => y(d['raw_signal']));

            // Add the line
            // WE ALSO SOMEHOW ALSO NEED TO SET THE CLIP PATH
            // BECAUSE IT IS IN THIS LINE THAT WE WILL ZOOM IN
            let raw_line_path = g.append("path")
            .datum(initSprSheet)
            .attr("class", "raw-line")
            .attr("fill", "none")
            .attr("stroke", "url(#raw-line-gradient)" )
            .attr("stroke-width", 2)
            .attr("clip-path", "url(#clip)")
            .attr("d", raw_line)
            .style("display", showRaw == true ? "block" : "none");

            let artifact = d3.line()
            .x(d => x(d['time']))
            .y(d => d['label'] == 1 ? y(max_signal): y(0));

            let artifact_path = g.append("path")
            .datum(initSprSheet)
            .attr("class", "artifact")
            .attr("fill", "none")
            .attr("stroke", "rgba(252, 36, 3, 0.25)")
            .attr("stroke-width", 1)
            .attr("clip-path", "url(#clip)")
            .attr("d", artifact);
            
            let clean_line_path, clean_line;
            if('new_signal' in initSprSheet[0]){
                // set the gradient for corrected signal
                g.append("linearGradient")
                .attr("class", "clean-line-gradient")
                .attr("id", "clean-line-gradient")
                .attr("gradientUnits", "userSpaceOnUse")
                .attr("x1", 0)
                .attr("y1", y(0))
                .attr("x2", 0)
                .attr("y2", y(max_signal))
                .selectAll("stop")
                .data([
                    {offset: "0%", color: "#f7a50c"},
                    {offset: "50%", color: "#f77e0c"},
                    {offset: "100%", color: "#e8340c"}
                ])
                .enter().append("stop")
                .attr("offset", (d) => d["offset"])
                .attr("stop-color", (d) => d["color"]);

                // define line generator for clean signal
                clean_line = d3.line()
                .x(d => x(d['time']))
                .y(d => y(d['new_signal']));

                // Add the line containing corrected signal
                clean_line_path = g.append("path")
                .datum(initSprSheet)
                .attr("class", "clean-line")
                .attr("fill", "none")
                .attr("stroke", "url(#clean-line-gradient)" )
                .attr("stroke-width", 2)
                .attr("clip-path", "url(#clip)")
                .attr("d", clean_line)
                .style("display", showCorrect == true ? "block" : "none");
            }

            let baseline, baseline_path, medium_stress, medium_stress_path, high_stress, high_stress_path;
            if('stress_level' in initSprSheet[0]){
                baseline = d3.line()
                .x(d => x(d['time']))
                .y(d => d['stress_level'] == 0 ? y(max_signal) : y(0));

                baseline_path = g.append("path")
                .datum(initSprSheet)
                .attr("class", "baseline")
                .attr("fill", "none")
                .attr("stroke", "rgb(255, 0, 93)")
                .attr("stroke-width", 1)
                .attr("clip-path", "url(#clip)")
                .attr("d", baseline);

                medium_stress = d3.line()
                .x(d => x(d['time']))
                .y(d => d['stress_level'] == 1 ? y(max_signal) : y(0));

                medium_stress_path = g.append("path")
                .datum(initSprSheet)
                .attr("class", "medium-stress")
                .attr("fill", "none")
                .attr("stroke", "rgb(93, 0, 255)")
                .attr("stroke-width", 1)
                .attr("clip-path", "url(#clip)")
                .attr("d", medium_stress);

                high_stress = d3.line()
                .x(d => x(d['time']))
                .y(d => d['stress_level'] == 2 ? y(max_signal) : y(0));

                high_stress_path = g.append("path")
                .datum(initSprSheet)
                .attr("class", "high-stress")
                .attr("fill", "none")
                .attr("stroke", "rgb(11, 0, 90)")
                .attr("stroke-width", 1)
                .attr("clip-path", "url(#clip)")
                .attr("d", high_stress); 
            }

            g.append("defs")
            .append("clipPath")
            .attr("id", "clip")
            .append("rect")
            .attr("width", width)
            .attr("height", height);

            // setup zoom object
            let zoom = d3.zoom()
            .scaleExtent([1 / 4, Infinity])
            .translateExtent([
                [-width, -Infinity],
                [2 * width, Infinity]
            ])
            .on("zoom", (event, datum) => {
                let new_x = event.transform.rescaleX(x);
                x_group.call(x_axis.scale(new_x));
                raw_line_path.attr("d", raw_line.x((d) => new_x(d['time'])));
                if('new_signal' in initSprSheet[0]){
                    clean_line_path.attr("d", clean_line.x((d) => new_x(d['time'])))
                }

                if('stress_level' in initSprSheet[0]){
                    baseline_path.attr("d", baseline.x((d) => new_x(d['time'])));
                    medium_stress_path.attr("d", medium_stress.x((d) => new_x(d['time'])));
                    high_stress_path.attr("d", high_stress.x((d) => new_x(d['time'])));
                }

                // for(let line of lines){
                //     line
                //     .attr("x1", x((d) => new_x(d['time'])))
                //     .attr("x2", x((d) => new_x(d['time'])));
                // }
                artifact_path.attr("d", artifact.x((d) => new_x(d['time'])))
            });

            zoom.translateExtent([
                [x(min_sec, -Infinity)],
                [x(max_sec, Infinity)]
            ]);

            let zoom_rect = svg.append("rect")
            .attr("width", width)
            .attr("height", height)
            .attr("fill", "none")
            .attr("pointer-events", "all")
            .call(zoom);

            zoom_rect.call(zoom.transform, d3.zoomIdentity);

        }
    }, [initSprSheet]);

    useEffect(() => {
        // get id or class of path object appended to g object containing the
        // raw signal or corrected signal
        let raw_line_path = d3.select('.raw-line');
        // if this is null due to the element not existing then
        // this line will not anyway run because of conditional chain
        raw_line_path?.style("display", showRaw == true ? "block" : "none");

        // apply same rule to clean line path object
        let clean_line_path = d3.select('.clean-line');
        clean_line_path?.style("display", showCorrect == true ? "block" : "none");

        let artifact_path = d3.select('.artifact');
        artifact_path?.style("display", showArt == true ? "block": "none");

        let baseline_path = d3.select('.baseline');
        baseline_path?.style("display", showStressLevels == true ? "block": "none");

        let medium_stress_path = d3.select('.medium-stress');
        medium_stress_path?.style("display", showStressLevels == true ? "block": "none");

        let high_stress_path = d3.select('.high-stress');
        high_stress_path?.style("display", showStressLevels == true ? "block": "none");

    }, [showCorrect, showArt, showRaw, showStressLevels])

    return (
        
        <div className={`spreadsheet-graph-container ${design}`} style={style} width={800} height={400}>
            {/* when spreadsheet is uploaded spreadsheets becomes are not null anymore
            but when another upload occurs and is cancelled spreadsheets becomes
            a list of length 0 
            
            when the .csv spreadsheet is uploaded we need someway to visualize
            the signals inside it. So we need to parse the uploaded .csv
            spreadsheet into some kind of dataframe
            */}
            <svg ref={svgRef} className="spreadsheet-graph">

            </svg>
        </div>
    );
}