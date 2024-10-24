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
        console.log("state updated");
        console.log(initSprSheet);
        console.log()

        // upon upload of user of .csv file or new .csv file
        // check also if there is already an existing g child
        // under the svg element, if there is remove it so it can
        // be replaced by new g element
        svg.selectAll("*").remove();
        
        // when initial spreadsheet is removed because of submit
        // and be replaced by final spreadsheet, final spreadsheet
        // will now replace this initial spreadsheet svg
        if(initSprSheet.length != 0){
            
            let max_signal = d3.max(initSprSheet, (row) => row['raw_signal']);
            let min_signal = d3.min(initSprSheet, (row) => row['raw_signal']);
            let min_sec = initSprSheet[0]['time'];
            let max_sec = initSprSheet[initSprSheet.length - 1]['time'];

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
            console.log(g);

            // x here is a callback function
            let x = d3.scaleLinear()
            .domain([min_sec, max_sec])
            .range([0, width]);

            // we create a g element which will draw the x-axis
            g.append('g')
            .attr("class", "x-axis")
            .attr('transform', `translate(0, ${height})`)
            .call(d3.axisBottom(x));
            
            // y here is also callback function
            let y = d3.scaleLinear()
            .domain([0, max_signal])
            .range([height, 0]);

            // we create a g element which will draw the y-axis
            g.append("g")
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
      
            // Add the line
            g.append("path")
            .datum(initSprSheet)
            .attr("fill", "none")
            .attr("stroke", "url(#line-gradient)" )
            .attr("stroke-width", 2)
            .attr("d", d3.line()
                .x((d) => x(d['time']))
                .y((d) => y(d['raw_signal']))
            );

        }else if(finalSprSheet.length != 0){
            console.log(finalSprSheet);

            let max_signal = d3.max(finalSprSheet, (row) => row['new_signal']);
            let min_signal = d3.min(finalSprSheet, (row) => row['new_signal']);
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
            g.append('g')
            .attr("class", "x-axis")
            .attr('transform', `translate(0, ${height})`)
            .call(d3.axisBottom(x));
            
            // y here is also callback function
            let y = d3.scaleLinear()
            .domain([0, max_signal])
            .range([height, 0]);

            // we create a g element which will draw the y-axis
            g.append("g")
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
                {offset: "0%", color: "#258a11"},
                {offset: "50%", color: "#08639c"},
                {offset: "100%", color: "#460699"}
            ])
            .enter().append("stop")
            .attr("offset", (d) => d["offset"])
            .attr("stop-color", (d) => d["color"]);
      
            // Add the line
            g.append("path")
            .datum(finalSprSheet)
            .attr("fill", "none")
            .attr("stroke", "url(#line-gradient)" )
            .attr("stroke-width", 2)
            .attr("d", d3.line()
                .x((d) => x(d['time']))
                .y((d) => y(d['new_signal']))
            );

        }else{
            console.log('spreadsheet input mounted');
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