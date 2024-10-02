import { useContext, useEffect, useRef } from "react";
import { ThemeContext } from "../contexts/ThemeContext";
import { DesignsContext } from "../contexts/DesignsContext";
import { FormInputsContext } from "../contexts/FormInputsContext";

import Papa from "papaparse";
import * as d3 from 'd3';

export default function SpreadSheetInput({ children, props }){
    // recall useRef() is akin in html to selecting an element via
    // the document tree object
    const svgRef = useRef();

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

    // based on the context provider of wrapped Form containing
    // all its states we use the state appropriate to the ImageInput
    // component and its setter to set from this component the state of
    // the form
    let { sprSheet, setSprSheet } = useContext(FormInputsContext);

    const handleUpload = (event) => {
        event.preventDefault();
        let reader = new FileReader();
        let file = event.target.files[0];
        reader.onload = (event) => {
            let csvToText = event.target.result;
            let output = csvToJSON(csvToText);
            setSprSheet(output);
        };
        reader.readAsText(file);
    }

    const csvToJSON = (csv) => {
        let lines = csv.split("\n");
        let result = [];
        let headers;
        // headers = lines[0].split(";");
        headers = ['time', 'raw_signal', 'clean_signal', 'label', 'auto_signal', 'pred_art', 'post_proc_pred_art'];
    
        for (let i = 1; i < lines.length; i++) {
            let obj = {};
    
            if(lines[i] == undefined || lines[i].trim() == "") {
                continue;
            }
    
            let words = lines[i].split(";");
            for(let j = 0; j < words.length; j++) {
                obj[headers[j].trim()] = words[j];
            }
    
            result.push(obj);
        }
        return result;
    }

    const toggle = design.includes('neomorphic') ? (event) => {
        console.log(event.target.classList);
        if(event.target.classList.contains('clicked')){
            event.target.classList.remove('clicked');
        }else{
            event.target.classList.add('clicked');
        }
    } : null;

    useEffect(() => {
        console.log("state updated");
        console.log(sprSheet);

        if(sprSheet.length != 0){
            let max_signal = d3.max(sprSheet, (row) => row['raw_signal']);
            let min_signal = d3.min(sprSheet, (row) => row['raw_signal']);
            let min_sec = sprSheet[0]['time'];
            let max_sec = sprSheet[sprSheet.length - 1]['time'];

            console.log(`max signal: ${max_signal}`);
            console.log(min_signal);
            console.log(max_sec);
            console.log(min_sec);

            // const width = 'clamp(500px, 75vw, 1260px)';
            // const height = '250px';
            const margin = {top: 10, right: 30, bottom: 30, left: 60, }
            const width = 768 - margin["left"] - margin["right"];
            const height = 486 - margin["top"] - margin["bottom"]; 

            const svg = d3.select(svgRef.current)
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", `translate(${margin["left"]}, ${margin["top"]})`);

            // x here is a callback function 
            let x = d3.scaleTime()
            .domain([min_sec, max_sec])
            .range([0, width]);

            svg.append('g')
            .attr('transform', `translate(0, ${height})`)
            .call(d3.axisBottom);
            
            // y here is also callback function
            let y = d3.scaleTime()
            .domain([min_signal, max_signal])
            .range([height, 0]);

            svg.append("g")
            .call(d3.axisLeft(y));
      
            // Set the gradient
            svg.append("linearGradient")
            .attr("id", "line-gradient")
            .attr("gradientUnits", "userSpaceOnUse")
            .attr("x1", 0)
            .attr("y1", y(0))
            .attr("x2", 0)
            .attr("y2", y(max_signal))
            .selectAll("stop")
            .data([
                {offset: "0%", color: "blue"},
                {offset: "100%", color: "red"}
            ])
            .enter().append("stop")
            .attr("offset", function(d) { return d.offset; })
            .attr("stop-color", function(d) { return d.color; });
      
          // Add the line
          svg.append("path")
            .datum(sprSheet)
            .attr("fill", "none")
            .attr("stroke", "url(#line-gradient)" )
            .attr("stroke-width", 2)
            .attr("d", d3.line()
                .x(function(d) { return x(d['time']) })
                .y(function(d) { return y(d['raw_signal']) })
            )

        }else{

            console.log('spreadsheet input mounted');
        }

        
        
        // // setting up x and y axes of the graph
        // const xScale = d3.scaleLinear()
        // .domain([0, sprSheet.length - 1])
        // .range([0, width])

        // const yScale = d3.scaleLinear()
        // .domain([0, height])
        // .range([height, 0])

        // const genScaledLine = d3.line()
        // .x((domain, x_i) => xScale(x_i))
        // .y(yScale)
        // .curve(d3.curveCardinal)

        // const xAxis = d3.axisBottom(xScale)
        // .ticks(sprSheet.length)
        // .tickFormat((i) => i + 1);

        // const yAxis = d3.axisLeft(yScale)
        // .ticks(5);

        // svg.append('g')
        // .call(xAxis)
        // .attr('transform', `translate(0, ${height})`);

        // svg.append('g')
        // .call(yAxis)

        // svg.selectAll('.line')
        // .data([sprSheet.map((row) => row['raw_signal'])])
        // .join('path')
        // .attr('d', (d) => genScaledLine(d))
        // .attr('fill', 'none')
        // .attr('stroke', 'black')
    }, [sprSheet]);

    return (
        
        <div className={`spreadsheet-upload-container ${design}`} style={style}>
            {/* when spreadsheet is uploaded spreadsheets becomes are not null anymore
            but when another upload occurs and is cancelled spreadsheets becomes
            a list of length 0 
            
            when the .csv spreadsheet is uploaded we need someway to visualize
            the signals inside it. So we need to parse the uploaded .csv
            spreadsheet into some kind of dataframe
            */}
            <svg ref={svgRef} className="spreadsheet-graph">

            </svg>
            <div className="spreadsheet-upload-field-wrapper">
                <label htmlFor="spreadsheet-upload" className="spreadsheet-upload-label">File</label>    
                <input 
                    type="file" 
                    accept="file/*" 
                    id="spreadsheet-upload" 
                    className={`spreadsheet-upload-field ${design}`} 
                    onChange={handleUpload}
                    onMouseDown={toggle} 
                    onMouseUp={toggle}
                />
            </div>
        </div>
    );
}