import { useContext, useEffect, useRef } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

import * as d3 from 'd3';

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

    let { initSprSheet, finalSprSheet } = useContext(FormInputsContext);

    // recall useRef() is akin in html to selecting an element via
    // the document tree object
    const svgRef = useRef();
    const svg = d3.select(svgRef.current);

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
            
            let max_signal = d3.max(initSprSheet, (row) => row['raw_signal']);
            let min_signal = d3.min(initSprSheet, (row) => row['raw_signal']);
            let min_sec = initSprSheet[0]['time'];
            let max_sec = initSprSheet[initSprSheet.length - 1]['time'];

            // console.log(`max signal: ${max_signal}`);
            // console.log(`min_signal: ${min_signal}`);
            // console.log(`max_sec: ${max_sec}`);
            // console.log(`min_sec: ${min_sec}`);

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

            // y_group.call(y_axis).select(".domain").remove();

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

            // set the gradient
            g.append("linearGradient")
            .attr("class", "line-gradient")
            .attr("id", "line-gradient")
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
      
            // define area
            let area = d3.area()
            .curve(d3.curveStepAfter)
            .y0(y(0))
            .y1((d) => y(d['new_signal']));

             // Add the line
            // WE ALSO SOMEHOW ALSO NEED TO SET THE CLIP PATH
            // BECAUSE IT IS IN THIS LINE THAT WE WILL ZOOM IN
            let area_path = g.append("path")
            .datum(initSprSheet)
            .attr("fill", "none")
            .attr("stroke", "url(#line-gradient)" )
            .attr("stroke-width", 2)
            .attr("d", d3.line()
                .x((d) => x(d['time']))
                .y((d) => y(d['raw_signal']))
            )
            .attr("clip-path", "url(#clip)")

            g.append("clipPath")
            .attr("id", "clip")
            .append("rect")
            .attr("width", width)
            .attr("height", height);

           


            // THIS IS THE LINE COLORED TURQOISE THAT USES THE FLIGHTS
            // FILE WHICH MEANS WE WOULD HAVE TO APPLY WHAT THIS PATH
            // OBJECT HAS TO OUR OWN PATH OBJECT THAT IS OUR TRUE LINE
            // let area_path = g.append("path")
            // .attr("clip-path", "url(#clip)")
            // .attr("fill", "none")
            // .attr("stroke", "#017c8d")
            // .attr("stroke-width", "2px");

            // g.append("clipPath")
            // .attr("id", "clip")
            // .append("rect")
            // .attr("width", width)
            // .attr("height", height

            // define zoom
            // scale extent is simply the amount d3 will have to 
            // zoom out or zoom in. .on is alsos an event listener
            // and will trigger 
            let zoom = d3.zoom()
            .scaleExtent([1 / 4, 8])
            .translateExtent([
                [-width, -Infinity],
                [2 * width, Infinity]
            ])
            .on("zoom", (event, datum) => {
                let new_x = event.transform.rescaleX(x);
                console.log(new_x)
                // var Gen = d3.line() 
                // .x((p) => p.xpoint) 
                // .y((p) => p.ypoint); 

                // d3.select("#gfg") 
                // .append("path") 
                // .attr("d", Gen(points)) 
                
                // // somehow we have to replicate this as d3.line().x().y() 
                // // will return a value the d attribute can take as a value 
                // .attr("d", d3.line()
                //     .x((d) => x(d['time']))
                //     .y((d) => y(d['raw_signal']))
                // )
                // and unfortunately area.x doesn't return such values
              
                // so the xAxis is used here
                // g.append("g")
                // .attr("transform", "translate(0," + height + ")")
                // .call(d3.axisBottom(x))
                // the only difference is instead of d3.axisBottom(x) or xAxis being passed
                // to .call we use the xAxis variable further to access .scale to pass
                // the new scaled x value returned from d3.event.transform.resecaleX(x)
                x_group.call(x_axis.scale(new_x));
                // error path attribute d: Expected number, "M-1014.345,NaNL-1014.343,Na…".
                area_path.attr(
                    "d",
                    area.x((d) => new_x(d["time"]))
                );
            });

            zoom.translateExtent([
                [x(min_sec, -Infinity)],
                [x(max_sec, Infinity)]
            ])

            let zoom_rect = svg.append("rect")
            .attr("width", width)
            .attr("height", height)
            .attr("fill", "none")
            .attr("pointer-events", "all")
            .call(zoom);

            // console.log(zoom.transform)
            // console.log(d3.zoomIdentity)

            console.log(zoom_rect);

        }else if(finalSprSheet.length != 0){
            // console.log(finalSprSheet);

            // get the 
            let max_raw_signal = d3.max(finalSprSheet, (row) => row['raw_signal']);
            let min_raw_signal = d3.min(finalSprSheet, (row) => row['raw_signal']);
            let max_clean_signal = d3.max(finalSprSheet, (row) => row['new_signal']);
            let min_clean_signal = d3.min(finalSprSheet, (row) => row['new_signal']);
            
            // get final max value between the max of the raw signal and the 
            // max of the clean signal
            let max_signal = Math.max(max_raw_signal, max_clean_signal);
            let min_signal = Math.max(min_raw_signal, min_clean_signal);

            let min_sec = finalSprSheet[0]['time'];
            let max_sec = finalSprSheet[finalSprSheet.length - 1]['time'];

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
            const g = svg
            .attr("width", width + margin.left + margin.right) // still is 768 since we add back the subtracted values from margin top and margin bottom
            .attr("height", height + margin.top + margin.bottom) // still is 486 since we add back the subtracted values from margin top and margin bottom
            .attr("viewBox", [
                0,
                0,
                (width + margin["left"] + margin["right"]),
                (height + margin["top"] + margin["bottom"])])
            .append("g")
            .attr("class", "cartesian-plane")
            .attr("transform", `translate(${margin["left"]}, ${margin["top"]})`); // this is the g element which draws the line
            console.log(g)

            // x here is a callback function
            let x = d3.scaleLinear()
            .domain([min_sec, max_sec])
            .range([0, width]);

            // we create a g element which will draw the x-axis
            let x_axis = g.append('g')
            .attr("class", "x-axis")
            .attr('transform', `translate(0, ${height})`)
            .call(d3.axisBottom(x));
            
            // y here is also callback function
            let y = d3.scaleLinear()
            .domain([0, max_signal])
            .range([height, 0]);

            // we create a g element which will draw the y-axis
            let y_axis = g.append("g")
            .attr("class", "y-axis")
            .call(d3.axisLeft(y));

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
                {offset: "0%", color: "#258a11"},
                {offset: "50%", color: "#08639c"},
                {offset: "100%", color: "#460699"}
            ])
            .enter().append("stop")
            .attr("offset", (d) => d["offset"])
            .attr("stop-color", (d) => d["color"]);
      
            // Add the line containing corrected signal
            g.append("path")
            .datum(finalSprSheet)
            .attr("fill", "none")
            .attr("stroke", "url(#clean-line-gradient)" )
            .attr("stroke-width", 2)
            .attr("d", d3.line()
                .x((d) => x(d['time']))
                .y((d) => y(d['new_signal']))
            );

            // set the gradient for raw signal
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
      
            // Add the line containing the raw signal
            g.append("path")
            .datum(finalSprSheet)
            .attr("fill", "none")
            .attr("stroke", "url(#raw-line-gradient)" )
            .attr("stroke-width", 2)
            .attr("d", d3.line()
                .x((d) => x(d['time']))
                .y((d) => y(d['raw_signal']))
            );

            // Set the zoom and Pan features: how much you can zoom, on which part, and what to do when there is a zoom
            let zoom = d3.zoom()
            .scaleExtent([2, 20])  // This control how much you can unzoom (x0.5) and zoom (x20)
            .extent([[0, 0], [width, height]])
            .on("zoom", updateChart);
            

            // This add an invisible rect on top of the chart area. This rect can recover pointer events: necessary to understand when the user zoom
            g.append("rect")
            .attr("width", width)
            .attr("height", height)
            .style("fill", "none")
            .style("pointer-events", "all")
            .attr('transform', `translate(${margin["top"]}, ${margin["left"]})`)
            .call(zoom);

            const updateChart = () => {
                
            }

        }else{
            // console.log('spreadsheet input mounted');
        }
        
    }, [initSprSheet, finalSprSheet]);

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